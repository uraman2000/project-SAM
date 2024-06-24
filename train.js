const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');

// Load and parse data from data.json
const rawData = fs.readFileSync('data.json');
const data = JSON.parse(rawData);

// Extract intents, patterns, and responses
const intents = data.intents;
let patterns = [];
let labels = [];
let responses = {};

intents.forEach(intent => {
  patterns = patterns.concat(intent.patterns);
  labels = labels.concat(Array(intent.patterns.length).fill(intent.tag));
  responses[intent.tag] = intent.responses;
});

// Manual text preprocessing - tokenization and encoding
const tokenizer = new Map();
let index = 0;

const tokenizedPatterns = patterns.map(pattern => {
  const words = pattern.toLowerCase().split(/[ ,.!?]/).filter(word => word.length > 0);
  const tokens = words.map(word => {
    if (!tokenizer.has(word)) {
      tokenizer.set(word, index++);
    }
    return tokenizer.get(word);
  });
  return tokens;
});

// Find maximum sequence length
const maxSequenceLength = Math.max(...tokenizedPatterns.map(tokens => tokens.length));

// Pad sequences to have consistent length
const paddedSequences = tokenizedPatterns.map(tokens => {
  const padLength = maxSequenceLength - tokens.length;
  return tokens.concat(new Array(padLength).fill(0));
});

// Convert labels to categorical
const labelEncoder = {};
labels.forEach((label, index) => {
  if (!labelEncoder[label]) {
    labelEncoder[label] = Object.keys(labelEncoder).length;
  }
});

const encodedLabels = labels.map(label => labelEncoder[label]);

// Define the model architecture
const model = tf.sequential();
model.add(tf.layers.dense({ units: 8, inputShape: [maxSequenceLength], activation: 'relu' }));
model.add(tf.layers.dense({ units: Object.keys(labelEncoder).length, activation: 'softmax' }));

// Compile the model
model.compile({
  optimizer: 'adam',
  loss: 'sparseCategoricalCrossentropy',
  metrics: ['accuracy']
});

// Train the model
const epochs = 100;
model.fit(
  tf.tensor2d(paddedSequences), tf.tensor1d(encodedLabels),
  { epochs, validationSplit: 0.2 }
).then((history) => {
  console.log('Training history:', history.history);
}).catch((error) => {
  console.error('Training failed:', error);
});

// Save the model
const MODEL_DIR = 'model';
if (!fs.existsSync(MODEL_DIR)) {
  fs.mkdirSync(MODEL_DIR);
}

model.save(`file://${MODEL_DIR}`)
  .then((saveResults) => {
    console.log('Model saved:', saveResults.modelArtifactsInfo);
  }).catch((saveError) => {
    console.error('Model saving failed:', saveError);
  });
