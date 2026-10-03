export const GET = () => {
  const items = [
    {
      title: "M7 site launch — all 8 pages live",
      date: "2026-10-03T00:00:00Z",
      description: "Full site launch with editorial design system, all 8 pages, pipeline integration, and prediction ledger.",
      link: "https://leadlaglab.com/changelog",
    },
    {
      title: "M6 prediction ledger + models",
      date: "2026-10-03T00:00:00Z",
      description: "Immutable prediction ledger with UUID + SHA-256 hash. ZeroModel, MomentumModel, and RidgeCV baselines.",
      link: "https://leadlaglab.com/changelog",
    },
    {
      title: "M5 evaluation engine",
      date: "2026-10-03T00:00:00Z",
      description: "IC/Spearman with Newey-West SEs, quintile returns, Fama-MacBeth, walk-forward + embargo, BH FDR correction.",
      link: "https://leadlaglab.com/changelog",
    },
    {
      title: "Pre-registration locked",
      date: "2026-10-03T00:00:00Z",
      description: "Study hypotheses, signal definitions, evaluation metrics, and failure criteria committed to PREREGISTRATION.md.",
      link: "https://leadlaglab.com/methodology#preregistration",
    },
  ];

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>Lead/Lag Lab Changelog</title>
    <link>https://leadlaglab.com</link>
    <description>Updates to the Lead/Lag Lab alt-data research project.</description>
    <language>en-us</language>
    <atom:link href="https://leadlaglab.com/feed.xml" rel="self" type="application/rss+xml"/>
    ${items
      .map(
        (item) => `
    <item>
      <title><![CDATA[${item.title}]]></title>
      <link>${item.link}</link>
      <pubDate>${new Date(item.date).toUTCString()}</pubDate>
      <guid>${item.link}#${item.date}</guid>
      <description><![CDATA[${item.description}]]></description>
    </item>`
      )
      .join("")}
  </channel>
</rss>`;

  return new Response(xml, {
    headers: {
      "Content-Type": "application/rss+xml; charset=utf-8",
      "Cache-Control": "public, max-age=3600",
    },
  });
};
