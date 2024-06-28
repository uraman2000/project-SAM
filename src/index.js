const express = require("express")

require("dotenv").config() // Load environment variables

const app = express()
const PORT = process.env.PORT || 3000

// Add /get test endpoint
app.get("/", (req, res) => {
  res.send("assss is working!");
});

app.get("/test", (req, res) => {
  res.send("Test endpoint is working!");
});
// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`)
})
