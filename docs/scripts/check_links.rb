#!/usr/bin/env ruby

require "pathname"
require "uri"

site = Pathname.new(ARGV.fetch(0, "_site")).expand_path
baseurl = ARGV.fetch(1, "/Condor").sub(%r{/$}, "")
failures = []

def candidates_for(path)
  return [path.join("index.html")] if path.to_s.end_with?(File::SEPARATOR)
  return [path] unless path.extname.empty?

  [path, path.join("index.html"), Pathname.new("#{path}.html")]
end

Dir.glob(site.join("**", "*.html")).sort.each do |html_name|
  html_path = Pathname.new(html_name)
  source = html_path.read
  source.scan(/(?:href|src)=["']([^"']+)["']/i).flatten.each do |raw_target|
    next if raw_target.empty? || raw_target.start_with?("#", "mailto:", "tel:", "data:", "javascript:")
    next if raw_target.match?(%r{\A[a-z][a-z0-9+.-]*://}i)

    target = raw_target.split(/[?#]/, 2).first
    next if target.nil? || target.empty?

    begin
      target = URI.decode_www_form_component(target)
    rescue ArgumentError
      failures << "#{html_path.relative_path_from(site)}: invalid URL encoding in #{raw_target}"
      next
    end

    if target.start_with?("/") && !baseurl.empty? &&
       target != baseurl && !target.start_with?("#{baseurl}/")
      failures << "#{html_path.relative_path_from(site)}: root-absolute path does not begin with #{baseurl}: #{raw_target}"
      next
    end

    resolved = if target.start_with?("/")
                 relative = target
                 relative = "/" if !baseurl.empty? && relative == baseurl
                 relative = relative.delete_prefix(baseurl) if !baseurl.empty? && relative.start_with?("#{baseurl}/")
                 site.join(relative.delete_prefix("/"))
               else
                 html_path.dirname.join(target).cleanpath
               end

    next if candidates_for(resolved).any?(&:exist?)

    failures << "#{html_path.relative_path_from(site)}: missing #{raw_target}"
  end
end

if failures.empty?
  puts "Internal links OK"
else
  warn failures.join("\n")
  abort "#{failures.length} broken internal link(s)"
end
