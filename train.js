const fs = require("fs")
const tf = require("@tensorflow/tfjs-node")
const natural = require("natural")

// Load and parse data from data.json
const rawData = fs.readFileSync("data.json")
const data = JSON.parse(rawData)

// Extract intents, patterns, and responses
const intents = data.intents
let patterns = []
let labels = []
let responses = {}

intents.forEach(intent => {
  patterns = patterns.concat(intent.patterns)
  labels = labels.concat(Array(intent.patterns.length).fill(intent.tag))
  responses[intent.tag] = intent.responses
})

// Advanced text preprocessing
const tokenizer = new natural.WordTokenizer()
const stemmer = natural.PorterStemmer

let tokenIndex = 1
const wordIndex = {}

// Tokenization and encoding
const tokenizedPatterns = patterns.map(pattern => {
  const words = tokenizer
    .tokenize(pattern.toLowerCase())
    .map(word => stemmer.stem(word))
  return words.map(word => {
    if (!wordIndex[word]) {
      wordIndex[word] = tokenIndex++
    }
    return wordIndex[word]
  })
})

// Find maximum sequence length
const maxSequenceLength = Math.max(
  ...tokenizedPatterns.map(tokens => tokens.length)
)

// Pad sequences to have consistent length
const paddedSequences = tokenizedPatterns.map(tokens => {
  const padLength = maxSequenceLength - tokens.length
  return tokens.concat(new Array(padLength).fill(0))
})

// Convert paddedSequences to a tensor
const inputTensor = tf.tensor2d(paddedSequences, [
  paddedSequences.length,
  maxSequenceLength,
])

// Convert labels to categorical
const labelIndex = {}
labels.forEach(label => {
  if (!labelIndex[label]) {
    labelIndex[label] = Object.keys(labelIndex).length
  }
})

const encodedLabels = labels.map(label => labelIndex[label])
const outputTensor = tf.tensor1d(encodedLabels, "int32").toFloat() // Cast to float32

// Define the model architecture
const model = tf.sequential()
model.add(
  tf.layers.embedding({
    inputDim: tokenIndex,
    outputDim: 64,
    inputLength: maxSequenceLength,
  })
)
model.add(tf.layers.lstm({ units: 64, returnSequences: true }))
model.add(tf.layers.lstm({ units: 32 }))
model.add(tf.layers.dropout({ rate: 0.5 }))
model.add(
  tf.layers.dense({
    units: Object.keys(labelIndex).length,
    activation: "softmax",
  })
)

// Compile the model
model.compile({
  optimizer: tf.train.adam(),
  loss: "sparseCategoricalCrossentropy",
  metrics: ["accuracy"],
})

// Manual Early Stopping
const epochs = 100
const validationSplit = 0.2
let bestValLoss = Infinity
let bestEpoch = -1
const patience = 5
let stopTraining = false

async function trainModel() {
  for (let epoch = 0; epoch < epochs; epoch++) {
    if (stopTraining) break

    const history = await model.fit(inputTensor, outputTensor, {
      epochs: 1,
      validationSplit: validationSplit,
    })

    const valLoss = history.history.val_loss[0]
    console.log(`Epoch ${epoch + 1}/${epochs} - val_loss: ${valLoss}`)

    if (valLoss < bestValLoss) {
      bestValLoss = valLoss
      bestEpoch = epoch
      await model.save(`file://model`)
    } else if (epoch - bestEpoch >= patience) {
      console.log(`Early stopping at epoch ${epoch + 1}`)
      stopTraining = true
    }
  }
}

trainModel()
  .then(() => {
    console.log("Training complete.")
  })
  .catch(error => {
    console.error("Training failed:", error)
  })

// Save the final model
const MODEL_DIR = "model"
if (!fs.existsSync(MODEL_DIR)) {
  fs.mkdirSync(MODEL_DIR)
}
const saveOptions = {
  fileSystem: true,
  quantizationBytes: 1, // Quantize weights to 1 byte (int8)
  writeFormat: tf.node.FORMAT_FILE,
}

model
  .save(`file://${MODEL_DIR}`, saveOptions)
  .then(saveResults => {
    console.log("Model saved:", saveResults.modelArtifactsInfo)
  })
  .catch(saveError => {
    console.error("Model saving failed:", saveError)
  })

// Function to predict the intent of a given input
async function predictIntent(input) {
  const words = tokenizer
    .tokenize(input.toLowerCase())
    .map(word => stemmer.stem(word))
  const inputTokens = words.map(word => wordIndex[word] || 0)
  const paddedInput = inputTokens.concat(
    new Array(maxSequenceLength - inputTokens.length).fill(0)
  )

  const inputTensor = tf.tensor2d([paddedInput], [1, maxSequenceLength])
  const prediction = model.predict(inputTensor)
  const predictedLabelIndex = prediction.argMax(-1).dataSync()[0]

  const predictedTag = Object.keys(labelIndex).find(
    key => labelIndex[key] === predictedLabelIndex
  )
  return predictedTag
}

// Test the prediction
predictIntent("Tell me about your work experience.").then(tag => {
  console.log("Predicted tag:", tag)
  console.log("Response:", responses[tag])
})
