const express = require("express")
const tf = require("@tensorflow/tfjs-node")
const fs = require("fs").promises
const cors = require("cors")
require("dotenv").config() // Load environment variables

const app = express()
const PORT = process.env.PORT || 3000

let model
async function loadModel() {
  try {
    model = await tf.loadLayersModel("file://model/model.json")
    console.log("Model loaded successfully")
  } catch (error) {
    console.error("Failed to load model:", error)
  }
}
loadModel()

// Load intents from data.json
let intents = []
async function loadIntents() {
  try {
    const data = await fs.readFile("data.json", "utf8")
    intents = JSON.parse(data).intents
    console.log("Intents loaded successfully")
  } catch (error) {
    console.error("Failed to load intents:", error)
  }
}
loadIntents()

// Function to classify input query and return response
function classifyQuery(query) {
  const words = query.toLowerCase().split(" ")
  let bestMatch = { tag: null, score: 0 }

  intents.forEach(intent => {
    intent.patterns.forEach(pattern => {
      const patternWords = pattern.toLowerCase().split(" ")
      let score = 0
      patternWords.forEach(word => {
        if (words.includes(word)) {
          score++
        }
      })
      if (score > bestMatch.score) {
        bestMatch = { tag: intent.tag, score: score }
      }
    })
  })

  if (bestMatch.tag) {
    const response = intents.find(
      intent => intent.tag === bestMatch.tag
    ).responses
    return response[Math.floor(Math.random() * response.length)] // Randomly choose a response
  } else {
    return "Sorry, I don't understand."
  }
}

const whitelist = process.env.CORS_WHITE_LIST.split(",") // Add your allowed origins here

// Configure CORS options
const corsOptions = {
  origin: function (origin, callback) {
    if (whitelist.indexOf(origin) !== -1 || !origin) {
      // Allow the request if the origin is in the whitelist or if there is no origin (e.g., mobile apps, curl requests)
      callback(null, true)
    } else {
      // Reject the request if the origin is not in the whitelist
      console.log("cors")
      callback(new Error("Not allowed by CORS"))
    }
  },
}

// Apply the CORS middleware to all routes
app.use(cors(corsOptions))

// Middleware to check API key
function checkApiKey(req, res, next) {
  const apiKey = req.headers["x-api-key"]

  if (!apiKey) {
    return res.status(403).json({ error: "No API key provided" })
  }

  if (apiKey !== process.env.API_KEY) {
    return res.status(403).json({ error: "Invalid API key" })
  }

  next()
}

// API endpoint for model inference and intent classification
app.post("/predict", checkApiKey, express.json(), async (req, res) => {
  try {
    const { question } = req.body

    // Check if question is provided
    if (!question) {
      return res
        .status(400)
        .json({ error: "Question is required in the request body." })
    }

    // Classify input query and get response (for intent-based response)
    const response = classifyQuery(question)

    res.json({ response }) // Return the response
  } catch (error) {
    console.error("Prediction failed:", error)
    res.status(500).json({ error: "Prediction failed" })
  }
})
app.get("/get", (req, res) => {
  res.send("Test endpoint is working!")
})
// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`)
})
