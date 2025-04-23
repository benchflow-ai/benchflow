# JFK Arena

A platform for comparing different AI models' responses to questions about JFK using RAG (Retrieval Augmented Generation). The platform enables users to evaluate and compare how different AI models interpret and respond to questions about historical events related to JFK.

## Features

- Interactive battle interface for comparing two AI models' responses to JFK-related questions
- Real-time voting system for response evaluation
- Dynamic leaderboard tracking model performance
- Support for multiple AI models including:
- Advanced RAG system using JFK historical documents
- Modern, responsive UI built with Next.js and Tailwind CSS
- Real-time data updates using PostgreSQL

## Prerequisites

- Node.js 18+ and npm
- Python 3.8+

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/jfkarena.git
cd jfkarena
```

2. Frontend setup:

```bash
cd frontend
cp .env.example .env
pnpm i
```

3. Backend setup:

```bash
cd backend
cp .env.example .env
python -m venv venv
source venv/bin/activate
brew install libmagic # On linux: apt-get install libmagic-dev file
pip install -r requirements.txt
```

4. Environment Configuration:

Frontend (.env):

```
NEXT_PUBLIC_API_URL=
JFK_ARENA_TOKEN=

BETTER_AUTH_SECRET=benchflow
NEXT_PUBLIC_BETTER_AUTH_URL=http://localhost:3000

DATABASE_URL=

GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=

```

Backend (.env):

```
OPENROUTER_API_KEY=
OPENAI_API_KEY=
DATABASE_URL=
JFK_ARENA_TOKEN=
```

## Development

1. Start the backend server:

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

2. Start the frontend development server:

```bash
cd frontend
npm run dev
```

The application will be available at `http://localhost:3000`

## Project Structure

```
jfkarena/
├── frontend/
│   ├── src/
│   │   ├── app/          # Next.js app router pages
│   │   ├── components/   # Reusable React components
│   │   └── lib/         # Utilities and configurations
│   ├── public/          # Static assets
│   ├── features/         # Modules
│   └── package.json
├── backend/
│   ├── main.py          # FastAPI application
│   ├── requirements.txt
│   └── cache/          # Vector store cache
└── README.md
```

## Deployment

The project is configured for deployment with:

- Frontend: Vercel
- Backend: Fly.io
- Database: NeonDB

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.
