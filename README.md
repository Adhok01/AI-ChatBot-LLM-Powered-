# AI ChatBot – LLM Powered

A modern AI-powered chatbot application built using Next.js, Ollama, and the Mistral 3B language model.  
The project delivers real-time conversational AI capabilities through a clean, responsive, and scalable web interface.

---

## Overview

This application enables users to interact with an AI assistant powered locally through Ollama using the Mistral 3B model.  
The project demonstrates the integration of modern frontend technologies with locally hosted Large Language Models (LLMs) for fast and efficient AI interactions.

The chatbot is designed with performance, simplicity, and scalability in mind while maintaining a professional user experience.

---

## Features

- AI-powered conversational chatbot
- Local LLM integration using Ollama
- Powered by the Mistral 3B model
- Real-time response generation
- Modern and responsive UI
- Built with scalable Next.js architecture
- Fast deployment support with Vercel

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Next.js | Full-stack React framework |
| React | Frontend UI library |
| TypeScript | Type-safe development |
| Tailwind CSS | Styling and responsive design |
| Ollama | Local LLM runtime |
| Mistral 3B | Language model |
| Vercel | Deployment platform |

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/Adhok01/AI-ChatBot-LLM-Powered-.git
```

### Navigate to the Project Directory

```bash
cd AI-ChatBot-LLM-Powered-
```

### Install Dependencies

```bash
npm install
```

---

## Install Ollama

Download and install Ollama:

https://ollama.com

Pull the Mistral 3B model:

```bash
ollama pull mistral
```

Start Ollama locally:

```bash
ollama serve
```

---

## Run the Development Server

```bash
npm run dev
```

Open the application:

```bash
http://localhost:3000
```

---

## Environment Variables

Create a `.env.local` file in the root directory if required:

```env
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Deployment

The application can be deployed using Vercel.

### Deployment Steps

1. Push the repository to GitHub
2. Import the repository into Vercel
3. Configure environment variables if needed
4. Deploy the application

---

## Common Deployment Issue

### 403 Forbidden Error

If the deployment succeeds but shows:

```bash
403 Forbidden
```

Possible causes:

- Missing environment variables
- Incorrect middleware configuration
- API access restrictions
- Ollama server not accessible in production

---

## Project Structure

```bash
├── app/                # Application routes and pages
├── components/         # Reusable UI components
├── public/             # Static assets
├── styles/             # Global styling
├── lib/                # Utility functions and API logic
├── package.json        # Dependencies and scripts
└── README.md
```

---

## Future Enhancements

- Persistent chat history
- Authentication system
- Multi-model support
- Voice assistant integration
- Database connectivity
- Streaming AI responses

---

## Author

**Adhokshaja Nagarhalli**

GitHub:  
https://github.com/Adhok01

---

## License

This project is licensed under the MIT License.
