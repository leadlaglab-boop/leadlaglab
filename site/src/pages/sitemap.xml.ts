import { loadUniverse } from "@/lib/data";

export const GET = () => {
  const universe = loadUniverse();

  const staticPages = [
    { url: "https://leadlaglab.com/", priority: "1.0", freq: "daily" },
    { url: "https://leadlaglab.com/signals", priority: "0.9", freq: "daily" },
    { url: "https://leadlaglab.com/signals/wiki-pageviews", priority: "0.8", freq: "weekly" },
    { url: "https://leadlaglab.com/signals/gdelt-news-tone", priority: "0.8", freq: "weekly" },
    { url: "https://leadlaglab.com/signals/google-trends", priority: "0.8", freq: "weekly" },
    { url: "https://leadlaglab.com/signals/sec-filings", priority: "0.8", freq: "weekly" },
    { url: "https://leadlaglab.com/stocks", priority: "0.9", freq: "weekly" },
    { url: "https://leadlaglab.com/explorer", priority: "0.8", freq: "daily" },
    { url: "https://leadlaglab.com/ledger", priority: "0.9", freq: "daily" },
    { url: "https://leadlaglab.com/methodology", priority: "0.7", freq: "monthly" },
    { url: "https://leadlaglab.com/changelog", priority: "0.7", freq: "daily" },
    { url: "https://leadlaglab.com/about", priority: "0.5", freq: "monthly" },
  ];

  const stockPages = universe.securities.map((s) => ({
    url: `https://leadlaglab.com/stocks/${s.ticker.toLowerCase()}`,
    priority: "0.5",
    freq: "daily",
  }));

  const allPages = [...staticPages, ...stockPages];
  const today = new Date().toISOString().slice(0, 10);

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${allPages
  .map(
    (p) => `  <url>
    <loc>${p.url}</loc>
    <lastmod>${today}</lastmod>
    <changefreq>${p.freq}</changefreq>
    <priority>${p.priority}</priority>
  </url>`
  )
  .join("\n")}
</urlset>`;

  return new Response(xml, {
    headers: {
      "Content-Type": "application/xml; charset=utf-8",
      "Cache-Control": "public, max-age=86400",
    },
  });
};
