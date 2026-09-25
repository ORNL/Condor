#pragma once

// Include necessary structs
#include "impl/Structs/Simdat.hpp"

// Include necessary functions
#include <cmath>

namespace Condor::impl{
    namespace Util{

        template<typename FloatType>
        inline void	AddToNodes(Nodes<FloatType>& nodes, const int_seg<FloatType>& seg) {
            // Convert the stored Gaussian coefficients to the base-2 exponential form once.
            const FloatType exp_scale = -static_cast<FloatType>(3.0) * std::log(static_cast<FloatType>(2.0));

            nodes.size++;
            nodes.xb.push_back(seg.xb);
            nodes.yb.push_back(seg.yb);
            nodes.zb.push_back(seg.zb);
            nodes.phix.push_back(exp_scale * seg.phix);	
            nodes.phiy.push_back(exp_scale * seg.phiy);	
            nodes.phiz.push_back(exp_scale * seg.phiz);	
            nodes.dtau.push_back(seg.dtau);
            nodes.expmod.push_back(exp_scale * static_cast<FloatType>(0.5) * std::log(seg.qmod * seg.qmod * seg.phix * seg.phiy * seg.phiz));
        }

        template<typename FloatType>
        inline void	CombineNodes(Nodes<FloatType>& nodes, const Nodes<FloatType>& nodes2) {
            nodes.size += nodes2.size;
            nodes.xb.insert(nodes.xb.end(), nodes2.xb.begin(), nodes2.xb.end());
            nodes.yb.insert(nodes.yb.end(), nodes2.yb.begin(), nodes2.yb.end());
            nodes.zb.insert(nodes.zb.end(), nodes2.zb.begin(), nodes2.zb.end());
            nodes.phix.insert(nodes.phix.end(), nodes2.phix.begin(), nodes2.phix.end());
            nodes.phiy.insert(nodes.phiy.end(), nodes2.phiy.begin(), nodes2.phiy.end());
            nodes.phiz.insert(nodes.phiz.end(), nodes2.phiz.begin(), nodes2.phiz.end());
            nodes.dtau.insert(nodes.dtau.end(), nodes2.dtau.begin(), nodes2.dtau.end());
            nodes.expmod.insert(nodes.expmod.end(), nodes2.expmod.begin(), nodes2.expmod.end()); 
        }

        template<typename FloatType>
        inline void	ClearNodes(Nodes<FloatType>& nodes) {
            nodes.size = 0;
            nodes.xb.clear();
            nodes.yb.clear();
            nodes.zb.clear();
            nodes.phix.clear();
            nodes.phiy.clear();
            nodes.phiz.clear();
            nodes.dtau.clear();
            nodes.expmod.clear();
        }
    }
}
