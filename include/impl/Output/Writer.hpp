#pragma once

#include "Definitions.hpp"
#include "impl/Structs/Simdat.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace Condor::impl::Out {

    enum class OutputMode {
        Surface2D,
        TopBottom2p5D,
        Volume3D,
        Volume3DFull
    };

    enum class OutputFormat {
        None,
        Csv
    };

    template<typename FloatType>
    struct WriterGrid {
        int x = 0;
        int y = 0;
        int z = 0;
        FloatType x0 = static_cast<FloatType>(0.0);
        FloatType y0 = static_cast<FloatType>(0.0);
        FloatType z0 = static_cast<FloatType>(0.0);
        FloatType dx = static_cast<FloatType>(1.0);
        FloatType dy = static_cast<FloatType>(1.0);
        FloatType dz = static_cast<FloatType>(1.0);

        int Columns() const { return x * y; }

        KOKKOS_INLINE_FUNCTION
        int Point(const int i, const int j, const int k) const {
            return i * (y * z) + j * z + k;
        }

        KOKKOS_INLINE_FUNCTION
        int Column(const int i, const int j) const {
            return i * y + j;
        }

        FloatType X(const int i) const { return x0 + static_cast<FloatType>(i) * dx; }
        FloatType Y(const int j) const { return y0 + static_cast<FloatType>(j) * dy; }
        FloatType Z(const int k) const { return z0 + static_cast<FloatType>(k) * dz; }
    };

    template<typename FloatType, typename View3DType>
    void CollectBottomSurface(
        const WriterGrid<FloatType>& grid,
        const View3DType& view_3d,
        const int_2D_deviceView& bottom_k,
        const Kokkos::View<FloatType**, layout, device_memory>& bottom_d) {
        Kokkos::parallel_for(
            "writer_collect_bottom",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {grid.x, grid.y}),
            KOKKOS_LAMBDA(const int i, const int j) {
                const int k = bottom_k(i, j);
                bottom_d(i, j) = (k < 0)
                    ? static_cast<FloatType>(0.0)
                    : static_cast<FloatType>(view_3d(i, j, k));
            });
    }

    template<typename FloatType>
    class Writer {
        private:
            using value_2d_device_view = Kokkos::View<FloatType**, layout, device_memory>;
            using value_2d_host_view = Kokkos::View<FloatType**, layout, host_transfer_memory>;
            using int_2d_host_view = Kokkos::View<int**, layout, host_transfer_memory>;

            struct FieldBase {
                explicit FieldBase(std::string field_name)
                    : name(std::move(field_name)) {}

                virtual ~FieldBase() = default;
                virtual void Collect(const Writer& writer) const = 0;
                virtual FloatType Value(const int i, const int j, const int k, const bool bottom) const = 0;

                std::string name;
            };

            template<typename ViewType>
            struct Field final : FieldBase {
                using value_type = typename ViewType::non_const_value_type;
                using host_mirror_type = Kokkos::View<typename ViewType::data_type, typename ViewType::array_layout, host_memory>;
                using unmanaged_3d_view = Kokkos::View<
                    value_type***,
                    Kokkos::LayoutRight,
                    device_memory,
                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

                explicit Field(const std::string& field_name, const ViewType& view_in)
                    : FieldBase(field_name), view(&view_in) {
                    static_assert(ViewType::rank == 1 || ViewType::rank == 3,
                        "Writer fields must be flattened rank-1 or rank-3 Kokkos views");
                    static_assert(std::is_same<value_type, FloatType>::value,
                        "Writer fields must use the same value type as Writer<FloatType>");
                }

                void Collect(const Writer& writer) const override {
                    grid_y = writer.grid_.y;
                    grid_z = writer.grid_.z;

                    switch (writer.mode_) {
                        case OutputMode::Surface2D:
                            volume_current = false;
                            CollectSurface_Top(writer);
                            break;
                        case OutputMode::TopBottom2p5D:
                            volume_current = false;
                            CollectSurface_Top(writer);
                            CollectSurface_Bot(writer);
                            break;
                        case OutputMode::Volume3D:
                        case OutputMode::Volume3DFull:
                            CollectVolume();
                            break;
                    }
                }

                FloatType Value(const int i, const int j, const int k, const bool bottom) const override {
                    if (volume_current) {
                        if constexpr (ViewType::rank == 1) {
                            return static_cast<FloatType>(volume_h(Point(i, j, k)));
                        }
                        else {
                            return static_cast<FloatType>(volume_h(i, j, k));
                        }
                    }

                    return bottom
                        ? static_cast<FloatType>(bottom_h(i, j))
                        : static_cast<FloatType>(top_h(i, j));
                }

                void CollectVolume() const {
                    EnsureVolumeStorage();
                    Kokkos::deep_copy(volume_h, *view);
                    volume_current = true;
                }

                void EnsureVolumeStorage() const {
                    bool needs_alloc = (volume_h.span() == 0);

                    if constexpr (ViewType::rank == 1) {
                        needs_alloc = needs_alloc ||
                            static_cast<int>(volume_h.extent(0)) != static_cast<int>(view->extent(0));
                    }
                    else {
                        needs_alloc = needs_alloc ||
                            static_cast<int>(volume_h.extent(0)) != static_cast<int>(view->extent(0)) ||
                            static_cast<int>(volume_h.extent(1)) != static_cast<int>(view->extent(1)) ||
                            static_cast<int>(volume_h.extent(2)) != static_cast<int>(view->extent(2));
                    }

                    if (needs_alloc) {
                        volume_h = Kokkos::create_mirror_view(*view);
                    }
                }

                int Point(const int i, const int j, const int k) const {
                    return i * (grid_y * grid_z) + j * grid_z + k;
                }

                void CollectSurface_Top(const Writer& writer) const {
                    EnsureTopStorage(writer);

                    const WriterGrid<FloatType> grid = writer.grid_;

                    if constexpr (ViewType::rank == 1) {
                        unmanaged_3d_view view_3d(view->data(), grid.x, grid.y, grid.z);
                        auto top_view = Kokkos::subview(view_3d, Kokkos::ALL(), Kokkos::ALL(), grid.z - 1);
                        Kokkos::deep_copy(top_h, top_view);
                    }
                    else {
                        auto top_view = Kokkos::subview(*view, Kokkos::ALL(), Kokkos::ALL(), grid.z - 1);
                        Kokkos::deep_copy(top_h, top_view);
                    }
                }

                void CollectSurface_Bot(const Writer& writer) const {
                    EnsureBottomStorage(writer);

                    const WriterGrid<FloatType> grid = writer.grid_;
                    if constexpr (ViewType::rank == 1) {
                        unmanaged_3d_view view_3d(view->data(), grid.x, grid.y, grid.z);
                        CollectBottomSurface(grid, view_3d, writer.bottom_k_d_, bottom_d);
                    }
                    else {
                        CollectBottomSurface(grid, *view, writer.bottom_k_d_, bottom_d);
                    }

                    Kokkos::deep_copy(bottom_h, bottom_d);
                }

                void EnsureTopStorage(const Writer& writer) const {
                    if (static_cast<int>(top_h.extent(0)) != writer.grid_.x ||
                        static_cast<int>(top_h.extent(1)) != writer.grid_.y) {
                        top_h = value_2d_host_view(
                            Kokkos::ViewAllocateWithoutInitializing("writer_top_h"),
                            writer.grid_.x,
                            writer.grid_.y);
                    }

                }

                void EnsureBottomStorage(const Writer& writer) const {
                    if (static_cast<int>(bottom_d.extent(0)) != writer.grid_.x ||
                        static_cast<int>(bottom_d.extent(1)) != writer.grid_.y) {
                        bottom_d = value_2d_device_view(
                            Kokkos::ViewAllocateWithoutInitializing("writer_bottom_d"),
                            writer.grid_.x,
                            writer.grid_.y);
                        bottom_h = value_2d_host_view(
                            Kokkos::ViewAllocateWithoutInitializing("writer_bottom_h"),
                            writer.grid_.x,
                            writer.grid_.y);
                    }
                }

                const ViewType* view = nullptr;
                mutable host_mirror_type volume_h;
                mutable value_2d_device_view bottom_d;
                mutable value_2d_host_view top_h;
                mutable value_2d_host_view bottom_h;
                mutable int grid_y = 0;
                mutable int grid_z = 0;
                mutable bool volume_current = false;
            };

        public:
            Writer(const Simdat<FloatType>& sim)
                : grid_{
                    sim.domain.xnum,
                    sim.domain.ynum,
                    sim.domain.znum,
                    sim.domain.xnum == 1 ? sim.domain.xmax : sim.domain.xmin,
                    sim.domain.ynum == 1 ? sim.domain.ymax : sim.domain.ymin,
                    sim.domain.znum == 1 ? sim.domain.zmax : sim.domain.zmin,
                    sim.domain.xnum == 1 ? static_cast<FloatType>(0.0) : sim.domain.xres,
                    sim.domain.ynum == 1 ? static_cast<FloatType>(0.0) : sim.domain.yres,
                    sim.domain.znum == 1 ? static_cast<FloatType>(0.0) : sim.domain.zres},
                  mode_(ParseMode(sim.settings.output.dims)),
                  format_(sim.settings.output.noOutput
                    ? OutputFormat::None
                    : ParseFormat(sim.settings.output.format)) {}

            Writer& SetOutputPrefix(const std::string& prefix) {
                output_prefix_ = prefix;
                return *this;
            }

            Writer& SetMode(const OutputMode mode) {
                mode_ = mode;
                return *this;
            }

            Writer& SetFormat(const OutputFormat format) {
                format_ = format;
                return *this;
            }

            template<typename ViewType>
            Writer& AddView(const std::string& name, const ViewType& view) {
                fields_.push_back(std::make_shared<Field<ViewType>>(name, view));
                return *this;
            }

            void ClearFields() {
                fields_.clear();
            }

            template<typename MaxDepthViewType>
            void Write(const int iter, const MaxDepthViewType& max_depth_by_column) const {
                string name;
                if (iter < 0) {
                    name = output_prefix_ + "_Final.csv";
                }
                else {
                    name = output_prefix_ + "_" + std::to_string(iter) + ".csv";
                }
                Write(name, max_depth_by_column);
            }

            template<typename MaxDepthViewType>
            void Write(const std::string& filename, const MaxDepthViewType& max_depth_by_column) const {
                if (format_ == OutputFormat::None) {
                    return;
                }
                if (format_ != OutputFormat::Csv) {
                    throw std::runtime_error("Unsupported Writer output format");
                }

                PrepareDepth(max_depth_by_column);
                for (const std::shared_ptr<FieldBase>& field : fields_) {
                    field->Collect(*this);
                }

                std::ofstream file(filename);
                if (!file.is_open()) {
                    throw std::runtime_error("Could not open output CSV file: " + filename);
                }

                file << std::setprecision(std::numeric_limits<FloatType>::max_digits10);
                WriteHeader(file);
                WriteRows(file);
            }

        private:
            static OutputMode ParseMode(const std::string& value) {
                if (value == "2D") {
                    return OutputMode::Surface2D;
                }
                if (value == "2.5D") {
                    return OutputMode::TopBottom2p5D;
                }
                if (value == "3D-Full") {
                    return OutputMode::Volume3DFull;
                }
                if (value == "3D") {
                    return OutputMode::Volume3D;
                }
                throw std::runtime_error("Unknown output dims: " + value);
            }

            static OutputFormat ParseFormat(const std::string& value) {
                if (value == "none") {
                    return OutputFormat::None;
                }
                if (value == ".csv") {
                    return OutputFormat::Csv;
                }
                throw std::runtime_error("Unknown output format: " + value);
            }

            template<typename MaxDepthViewType>
            void PrepareDepth(const MaxDepthViewType& max_depth_by_column) const {
                static_assert(MaxDepthViewType::rank == 1 || MaxDepthViewType::rank == 2,
                    "Max-depth view must be rank-1 columns or rank-2 (x,y)");

                if (mode_ == OutputMode::Volume3DFull) {
                    return;
                }

                if (static_cast<int>(bottom_k_d_.extent(0)) != grid_.x ||
                    static_cast<int>(bottom_k_d_.extent(1)) != grid_.y) {
                    bottom_k_d_ = int_2D_deviceView(
                        Kokkos::ViewAllocateWithoutInitializing("writer_bottom_k_d"),
                        grid_.x,
                        grid_.y);
                    bottom_k_h_ = int_2d_host_view(
                        Kokkos::ViewAllocateWithoutInitializing("writer_bottom_k_h"),
                        grid_.x,
                        grid_.y);
                }

                max_depth_h_.assign(static_cast<std::size_t>(grid_.Columns()), 0);

                auto depth_h = Kokkos::create_mirror_view(max_depth_by_column);
                Kokkos::deep_copy(depth_h, max_depth_by_column);

                for (int i = 0; i < grid_.x; ++i) {
                    for (int j = 0; j < grid_.y; ++j) {
                        const int column = grid_.Column(i, j);
                        const int raw_depth = ReadDepth(depth_h, i, j, column);
                        const int clipped_depth = std::max(0, std::min(raw_depth, grid_.z));

                        max_depth_h_[static_cast<std::size_t>(column)] = clipped_depth;
                        bottom_k_h_(i, j) = clipped_depth > 0 ? grid_.z - clipped_depth : -1;
                    }
                }

                Kokkos::deep_copy(bottom_k_d_, bottom_k_h_);
            }

            template<typename DepthHostViewType>
            int ReadDepth(const DepthHostViewType& depth_h, const int i, const int j, const int column) const {
                if constexpr (DepthHostViewType::rank == 1) {
                    if (column >= static_cast<int>(depth_h.extent(0))) {
                        return 0;
                    }
                    return depth_h(column);
                }
                else {
                    if (i >= static_cast<int>(depth_h.extent(0)) ||
                        j >= static_cast<int>(depth_h.extent(1))) {
                        return 0;
                    }
                    return depth_h(i, j);
                }
            }

            bool ColumnWasMelted(const int i, const int j) const {
                if (mode_ == OutputMode::Volume3DFull) {
                    return true;
                }
                return max_depth_h_[static_cast<std::size_t>(grid_.Column(i, j))] > 0;
            }

            int MeltedDepth(const int i, const int j) const {
                if (mode_ == OutputMode::Volume3DFull) {
                    return grid_.z;
                }
                return max_depth_h_[static_cast<std::size_t>(grid_.Column(i, j))];
            }

            void WriteHeader(std::ofstream& file) const {
                if (mode_ == OutputMode::Surface2D) {
                    file << "x,y";
                }
                else if (mode_ == OutputMode::TopBottom2p5D) {
                    file << "x,y,z,surface";
                }
                else {
                    file << "x,y,z";
                }

                for (const std::shared_ptr<FieldBase>& field : fields_) {
                    file << "," << field->name;
                }
                file << "\n";
            }

            void WriteRows(std::ofstream& file) const {
                for (int i = 0; i < grid_.x; ++i) {
                    for (int j = 0; j < grid_.y; ++j) {
                        if (!ColumnWasMelted(i, j)) {
                            continue;
                        }

                        if (mode_ == OutputMode::Surface2D) {
                            WriteSurfaceRow(file, i, j, grid_.z - 1, false);
                        }
                        else if (mode_ == OutputMode::TopBottom2p5D) {
                            WriteSurfaceRow(file, i, j, grid_.z - 1, false);
                            WriteSurfaceRow(file, i, j, grid_.z - MeltedDepth(i, j), true);
                        }
                        else {
                            const int depth = MeltedDepth(i, j);
                            for (int d = 0; d < depth; ++d) {
                                WriteVolumeRow(file, i, j, grid_.z - 1 - d);
                            }
                        }
                    }
                }
            }

            void WriteSurfaceRow(std::ofstream& file, const int i, const int j, const int k, const bool bottom) const {
                file << grid_.X(i) << "," << grid_.Y(j);
                if (mode_ == OutputMode::TopBottom2p5D) {
                    file << "," << grid_.Z(k) << "," << (bottom ? "bottom" : "top");
                }
                for (const std::shared_ptr<FieldBase>& field : fields_) {
                    file << "," << field->Value(i, j, k, bottom);
                }
                file << "\n";
            }

            void WriteVolumeRow(std::ofstream& file, const int i, const int j, const int k) const {
                file << grid_.X(i) << "," << grid_.Y(j) << "," << grid_.Z(k);
                for (const std::shared_ptr<FieldBase>& field : fields_) {
                    file << "," << field->Value(i, j, k, false);
                }
                file << "\n";
            }

            WriterGrid<FloatType> grid_;
            OutputMode mode_ = OutputMode::Volume3D;
            OutputFormat format_ = OutputFormat::Csv;
            std::string output_prefix_ = "output";
            std::vector<std::shared_ptr<FieldBase>> fields_;

            mutable int_2D_deviceView bottom_k_d_;
            mutable int_2d_host_view bottom_k_h_;
            mutable std::vector<int> max_depth_h_;
    };
}
