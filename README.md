# AI ChatBot – LLM Powered

A modern AI-powered chatbot application built with Next.js and Large Language Models (LLMs).  
The project delivers an interactive conversational experience with a clean user interface, scalable architecture, and seamless deployment workflow.

---

## Overview

This application enables users to interact with an AI assistant in real time using advanced language models. It is designed with performance, responsiveness, and developer experience in mind.

The project demonstrates the integration of modern frontend technologies with AI APIs to create an intelligent and production-ready chatbot platform.

---

## Features

- AI-powered conversational interface
- Real-time response generation
- Responsive and modern UI
- Scalable Next.js application structure
- API integration for LLM-based responses
- Optimized deployment with Vercel
- Clean and maintainable codebase

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Next.js | Full-stack React framework |
| React | Frontend UI library |
| TypeScript | Type-safe development |
| Tailwind CSS | UI styling |
| OpenAI API | AI response generation |
| Vercel | Hosting & deployment |

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

### Run the Development Server

```bash
npm run dev
```

The application will be available at:

```bash
http://localhost:3000
```

---

## Environment Variables

Create a `.env.local` file in the root directory and configure the following variables:

```env
OPENAI_API_KEY=your_openai_api_key
NEXTAUTH_SECRET=your_random_secret
NEXTAUTH_URL=http://localhost:3000
```

---

## Deployment

The application is optimized for deployment on Vercel.

### Deployment Steps

1. Push the repository to GitHub
2. Import the project into Vercel
3. Configure environment variables
4. Deploy the application

---

## Common Deployment Issue

### 403 Forbidden Error

If the deployment succeeds but the application displays:

```bash
403 Forbidden
```

Ensure that:

- Required environment variables are configured correctly
- Authentication settings are properly initialized
- Middleware configuration is valid
- API keys are active and accessible

After updating the configuration, redeploy the project.

---

## Project Structure

```bash
├── app/                # Application routes and pages
├── components/         # Reusable UI components
├── public/             # Static assets
├── styles/             # Global styling
├── lib/                # Utility functions and configurations
├── package.json        # Project metadata and dependencies
└── README.md
```

---

## Future Enhancements

- Persistent chat history
- User authentication system
- Multi-model AI integration
- Voice interaction support
- Database integration
- Advanced prompt customization

---

## Author

**Adhokshaja Nagarhalli**

GitHub:  
https://github.com/Adhok01

---

## License

This project is licensed under the MIT License.
