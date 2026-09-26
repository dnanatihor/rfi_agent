/** @type {import('next').NextConfig} */
const nextConfig = {
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  async rewrites() {
    return [{ source: "/backend/:path*", destination: "http://127.0.0.1:8000/:path*" }];
  },
};

export default nextConfig;
