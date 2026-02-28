# What we'll do

Journalists frequently struggle with the mountains of messy data piled up by our periphrastic society. This vast and verbose corpus contains entries as diverse as the long-hand entries in police reports, the arcane legalese of legislative bills and the seemingly endless stream of social media posts.

Understanding and analyzing this data is critical to the job but can be time-consuming and inefficient. Computers can help by automating away the drudgery of sorting through blocks of text, extracting key details and flagging unusual patterns.

```{raw} html
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<div id="graphic" style="margin: 10px 0; width: 100%;"></div>
<script>
const width = 900;
const height = 400;

const categories = [
  { label: "Restaurant", color: "#43a047", light: "#e8f5e9", y: 100 },
  { label: "Bar",        color: "#e65100", light: "#fff3e0", y: 150 },
  { label: "Hotel",      color: "#1565c0", light: "#e3f2fd", y: 200 },
  { label: "Other",      color: "#7b1fa2", light: "#f3e5f5", y: 250 },
  { label: "Unknown",    color: "#d32f2f", light: "#ffebee", y: 300 },
];

const svg = d3.select("#graphic")
  .append("svg")
  .attr("viewBox", `0 0 ${width} ${height}`)
  .attr("preserveAspectRatio", "xMidYMid meet")
  .style("width", "100%")
  .style("height", "auto");

// Source pipes on the left (3 pipes at different heights)
const pipeX = 0;
const pipeW = 80;
const pipeH = 40;
const pipePositions = [
  { y: 100 },   // top pipe
  { y: 150 },   // upper-middle pipe
  { y: 200 },   // center pipe
  { y: 250 },   // lower-middle pipe
  { y: 300 }    // bottom pipe
];

pipePositions.forEach(pipe => {
  // Pipe body - lighter gray to distinguish from funnel
  svg.append("rect")
    .attr("x", pipeX)
    .attr("y", pipe.y - pipeH / 2)
    .attr("width", pipeW)
    .attr("height", pipeH)
    .attr("rx", 6)
    .attr("fill", "#888");

  // Pipe opening (darker inner part)
  svg.append("rect")
    .attr("x", pipeW - 15)
    .attr("y", pipe.y - pipeH / 2 + 6)
    .attr("width", 15)
    .attr("height", pipeH - 12)
    .attr("rx", 3)
    .attr("fill", "#666");
});

// Ball layer BEHIND classifier (for incoming balls)
const ballLayerBehind = svg.append("g");

// Classifier funnel shape - classic trapezoid funnel
const classifierX = 350;
const classifierW = 180;
const funnelWideH = 260;  // wide opening height (left side)
const funnelNarrowH = 80; // narrow exit height (right side)
const classifierMidY = height / 2;

// Define gradient for 3D effect
const defs = svg.append("defs");

// Main funnel gradient (top to bottom shading)
const funnelGradient = defs.append("linearGradient")
  .attr("id", "funnelGradient")
  .attr("x1", "0%")
  .attr("y1", "0%")
  .attr("x2", "0%")
  .attr("y2", "100%");

funnelGradient.append("stop")
  .attr("offset", "0%")
  .attr("stop-color", "#4a4a4a");

funnelGradient.append("stop")
  .attr("offset", "50%")
  .attr("stop-color", "#2c2c2c");

funnelGradient.append("stop")
  .attr("offset", "100%")
  .attr("stop-color", "#1a1a1a");

// Drop shadow filter
const shadow = defs.append("filter")
  .attr("id", "dropShadow")
  .attr("x", "-20%")
  .attr("y", "-20%")
  .attr("width", "140%")
  .attr("height", "140%");

shadow.append("feDropShadow")
  .attr("dx", "3")
  .attr("dy", "3")
  .attr("stdDeviation", "4")
  .attr("flood-color", "rgba(0,0,0,0.4)");

// Simple trapezoid: wide on left, narrow on right
const funnelPath = `
  M ${classifierX} ${classifierMidY - funnelWideH / 2}
  L ${classifierX + classifierW} ${classifierMidY - funnelNarrowH / 2}
  L ${classifierX + classifierW} ${classifierMidY + funnelNarrowH / 2}
  L ${classifierX} ${classifierMidY + funnelWideH / 2}
  Z
`;

// Main funnel body with gradient and shadow
svg.append("path")
  .attr("d", funnelPath)
  .attr("fill", "url(#funnelGradient)")
  .attr("filter", "url(#dropShadow)");

// Add highlight edge on top
svg.append("line")
  .attr("x1", classifierX)
  .attr("y1", classifierMidY - funnelWideH / 2)
  .attr("x2", classifierX + classifierW)
  .attr("y2", classifierMidY - funnelNarrowH / 2)
  .attr("stroke", "#666")
  .attr("stroke-width", 2);

// Add dark edge on bottom
svg.append("line")
  .attr("x1", classifierX)
  .attr("y1", classifierMidY + funnelWideH / 2)
  .attr("x2", classifierX + classifierW)
  .attr("y2", classifierMidY + funnelNarrowH / 2)
  .attr("stroke", "#111")
  .attr("stroke-width", 2);

// Funnel entrance opening (left side - darker inner edge like input pipes)
svg.append("rect")
  .attr("x", classifierX)
  .attr("y", classifierMidY - funnelWideH / 2 + 8)
  .attr("width", 15)
  .attr("height", funnelWideH - 16)
  .attr("rx", 3)
  .attr("fill", "#1a1a1a")
  .attr("opacity", 0.7);

// Funnel exit opening (right side - darker inner edge like output pipes)
svg.append("rect")
  .attr("x", classifierX + classifierW - 15)
  .attr("y", classifierMidY - funnelNarrowH / 2 + 6)
  .attr("width", 15)
  .attr("height", funnelNarrowH - 12)
  .attr("rx", 3)
  .attr("fill", "#1a1a1a")
  .attr("opacity", 0.7);

// Variables for animation reference
const classifierH = funnelWideH;
const classifierY = classifierMidY - funnelWideH / 2;

// Ball layer IN FRONT of classifier (for outgoing balls)
const ballLayerFront = svg.append("g");

// Output pipes on the right - extend off screen (same width as input pipes)
const funnelX = 820;
const funnelW = 80;   // same width as input pipes (pipeW)
const funnelH = 40;

categories.forEach(cat => {
  // Pipe body extending off-screen - semi-transparent to see balls inside
  svg.append("rect")
    .attr("x", funnelX)
    .attr("y", cat.y - funnelH / 2)
    .attr("width", funnelW)
    .attr("height", funnelH)
    .attr("rx", 6)
    .attr("fill", cat.light)
    .attr("stroke", cat.color)
    .attr("stroke-width", 2)
    .attr("opacity", 0.6);

  // Pipe opening (darker inner part on left side)
  svg.append("rect")
    .attr("x", funnelX)
    .attr("y", cat.y - funnelH / 2 + 6)
    .attr("width", 15)
    .attr("height", funnelH - 12)
    .attr("rx", 3)
    .attr("fill", cat.color)
    .attr("opacity", 0.3);
});

// Counters
const counters = {};
categories.forEach(cat => { counters[cat.label] = 0; });

const counterTexts = {};
categories.forEach(cat => {
  counterTexts[cat.label] = svg.append("text")
    .attr("x", funnelX + funnelW / 2)
    .attr("y", cat.y + 6)
    .attr("text-anchor", "middle")
    .attr("fill", cat.color)
    .attr("font-size", 18)
    .attr("font-weight", 600)
    .text("0");
});

function launchBall() {
  const cat = categories[Math.floor(Math.random() * categories.length)];
  const radius = 7 + Math.random() * 4;
  const classifierMidY = height / 2;
  const classifierExitX = classifierX + classifierW;
  const endX = funnelX + 20;

  // Pick a random pipe to launch from
  const pipe = pipePositions[Math.floor(Math.random() * pipePositions.length)];

  // Ball starts from inside the selected pipe
  const ball = ballLayerBehind.append("circle")
    .attr("cx", pipeW - 5)
    .attr("cy", pipe.y)
    .attr("r", radius)
    .attr("fill", cat.color)
    .attr("opacity", 0.85);

  // Calculate consistent speed (pixels per ms) based on left side
  const leftDistance = classifierX + 10 - (pipeW - 5);
  const leftDuration = 1200;
  const speed = leftDistance / leftDuration;

  // Calculate right side distances
  const rightPhase3Distance = endX - (classifierExitX - 10);
  const rightPhase4Distance = (funnelX + funnelW + 20) - endX;

  // Calculate durations to match speed
  const phase3Duration = rightPhase3Distance / speed;
  const phase4Duration = rightPhase4Distance / speed;

  // Phase 1: fly straight toward classifier (keeping same y position)
  ball.transition()
    .duration(leftDuration)
    .ease(d3.easeLinear)
    .attr("cx", classifierX + 10)
    .attr("cy", pipe.y)
    .on("end", function() {
      // Move ball to FRONT layer when it exits classifier
      const node = d3.select(this);
      const cx = +node.attr("cx");
      const cy = +node.attr("cy");
      const r = +node.attr("r");
      const fill = node.attr("fill");
      node.remove();

      // Create new ball in front layer at exit point
      const ballFront = ballLayerFront.append("circle")
        .attr("cx", classifierExitX - 10)
        .attr("cy", classifierMidY + (cat.y - classifierMidY) * 0.2)
        .attr("r", r)
        .attr("fill", fill)
        .attr("opacity", 0.85);

      // Phase 3: fan out to correct funnel (same speed as left side)
      ballFront.transition()
        .duration(phase3Duration)
        .ease(d3.easeLinear)
        .attr("cx", endX)
        .attr("cy", cat.y)
        .transition()
        // Phase 4: slide into funnel and fade (same speed)
        .duration(phase4Duration)
        .ease(d3.easeLinear)
        .attr("cx", funnelX + funnelW + 20)
        .attr("opacity", 0)
        .on("end", function () {
          d3.select(this).remove();
          counters[cat.label]++;
          counterTexts[cat.label].text(counters[cat.label]);
        });
    });
}

// Staggered launch - more balls with shorter delays
function scheduleNext() {
  const delay = 80 + Math.random() * 150;
  setTimeout(() => {
    launchBall();
    scheduleNext();
  }, delay);
}

// Launch more initial balls
for (let i = 0; i < 15; i++) {
  setTimeout(launchBall, i * 100);
}
scheduleNext();
</script>
```

A common goal in this work is to classify text into categories. For example, you might want to sort a collection of emails as “spam” and “not spam” or identify corporate filings that suggest a company is about to go bankrupt.

Traditional techniques for classifying text — like keyword searches and [regular expressions](https://en.wikipedia.org/wiki/Regular_expression) — can be brittle and error-prone. Traditional machine-learning models can be more flexible, though they require large amounts of human training, a high level of computer programming expertise and often yield unimpressive results.

Large-language models offer a better deal. We will demonstrate how you can use them to get superior results with less effort.

## The LLM advantage

![A rotating set of LLM providers](_static/llm-chatbots.gif)

A [large-language model](https://en.wikipedia.org/wiki/Large_language_model) is an artificial intelligence system capable of understanding and generating human language due to its extensive training on vast amounts of text. These systems are commonly referred to by the acronym LLM. The most prominent examples include OpenAI’s GPT, Google’s Gemini and Anthropic’s Claude. There are many others, including numerous open-source options.

While these LLMs are most famous for their ability to converse with humans as chatbots, they can also perform a wide range of language processing tasks, including text classification, summarization and translation.

Unlike traditional machine-learning models, LLMs do not require users to provide pre-prepared training data to perform a specific task. Instead, LLMs can be prompted with a broad description of their goals and a few examples of rules they should follow. This approach is called prompting.

After parsing the user's request, the LLMs will generate responses informed by the massive amount of information they contain. That deep knowledge can be especially beneficial when dealing with large datasets that are difficult for humans to process on their own.

LLMs also do not require the user to understand machine-learning concepts, like vectorization or Bayesian statistics, or to write complex code to train and evaluate the model. Instead, users can submit prompts in plain language, which the model will use to generate responses. This makes it easier for journalists to experiment with different approaches and quickly iterate on their work.

## Our example case

To show the power of this approach, we’ll focus on a specific data set: campaign expenditures.

Candidates for office must disclose the money they spend on everything from pizza to private jets. Tracking their spending can reveal patterns and lead to important stories.

It’s no easy task. Each election cycle, thousands of candidates log their transactions into the public databases where spending is disclosed. The flood of filings results in too much data for anyone to examine on their own. To make matters worse, campaigns often use vague or misleading descriptions of their spending, making their activities difficult to understand.

For instance, it wasn’t until after his 2022 election to Congress that [journalists discovered](https://www.nytimes.com/2022/12/29/nyregion/george-santos-campaign-finance.html) that Rep. George Santos of New York had spent thousands of campaign dollars on questionable and potentially illegal expenses. While much of his shady spending was publicly disclosed, it was largely overlooked in the run-up to election day.

[![Santos story](_static/santos.png)](https://www.nytimes.com/2022/12/29/nyregion/george-santos-campaign-finance.html)

Inspired by this scoop, we will create a classifier that can scan the expenditures logged in campaign finance reports and identify those that may be newsworthy.

[![CCDC](_static/ccdc.png)](https://californiacivicdata.org/)

We will draw data from The Golden State, where the California Civic Data Coalition developed a clean, structured version of the statehouse's disclosure data, which tracks the spending of candidates for state legislature, governor and other statewide offices. The records are now stored by the [Big Local News](https://biglocalnews.org/), an open-source journalism project hosted by Stanford University.
