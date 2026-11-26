<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://ai.google.dev/static/site-assets/images/share-ais-513315318.png" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/f286b498-fd8e-458b-8409-3f02512a3ef0

## Netlify deployment

Netlify uses `bun run build` and publishes `dist`. The checked-in frontend is a standalone starter flashcard deck that does not require credentials or a backend. The original AI Studio frontend, its assets, and its server-side features are not part of this static deployment.

To run the standalone frontend locally, install dependencies with `bun install`, then run `bunx vite`.

## Run the original server locally

These instructions require the complete AI Studio export, including the source modules imported by `server.ts`.

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`
