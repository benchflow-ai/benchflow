/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  images: {
    remotePatterns: [],
    minimumCacheTTL: 60,
    dangerouslyAllowSVG: false,
  },
  logging: {
    fetches: {
      fullUrl: false,
    },
  },
  headers: async () => {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()',
          },
          {
            key: 'Content-Security-Policy',
            // eslint-disable-next-line node/prefer-global/process
            value: `default-src 'self'; script-src 'self' 'unsafe-eval' 'unsafe-inline' https://vercel.live${process.env.VERCEL_BRANCH_URL ? ` https://${process.env.VERCEL_BRANCH_URL}` : ''}; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' https://openrouter.ai/api/ https://vercel.live${process.env.VERCEL_BRANCH_URL ? ` https://${process.env.VERCEL_BRANCH_URL}` : ''};`,
          },
        ],
      },
    ];
  },
  async rewrites() {
    return [
      {
        source: "/ingest/static/:path*",
        destination: "https://us-assets.i.posthog.com/static/:path*",
      },
      {
        source: "/ingest/:path*",
        destination: "https://us.i.posthog.com/:path*",
      },
      {
        source: "/ingest/decide",
        destination: "https://us.i.posthog.com/decide",
      },
    ];
  },
  // This is required to support PostHog trailing slash API requests
  skipTrailingSlashRedirect: true,
  experimental: {
    optimizePackageImports: ['@radix-ui/react-progress', '@radix-ui/react-select', '@radix-ui/react-separator', '@radix-ui/react-slot', '@radix-ui/react-tabs'],
    serverActions: {
      bodySizeLimit: '2mb',
    },
  },
  serverRuntimeConfig: {
    // Increase the timeout for server functions
    api: {
      bodyParser: {
        sizeLimit: '2mb',
      },
      responseLimit: '60s',
    },
  },
};

export default nextConfig;
