async function analyzeSentiment() {
    const text = document.getElementById('review-input').value;
    if (!text.trim()) {
        alert("Please enter some text!");
        return;
    }

    // Show loading state
    const btn = document.querySelector('.btn');
    btn.textContent = "Analyzing...";
    btn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error("Error:", error);
        alert("Something went wrong with the analysis.");
    } finally {
        btn.textContent = "Analyze Sentiment";
        btn.disabled = false;
    }
}

function displayResults(data) {
    const resultSection = document.getElementById('result-section');
    const sentimentValue = document.getElementById('sentiment-value');
    const confidenceValue = document.getElementById('confidence-value');
    const gaugeFill = document.getElementById('gauge-fill');
    const textHighlighted = document.getElementById('text-highlighted');

    resultSection.style.display = 'block';

    // Update Sentiment
    sentimentValue.textContent = data.sentiment;
    sentimentValue.className = 'sentiment-value ' + data.sentiment.toLowerCase();

    // Update Confidence & Gauge
    const confidence = (data.confidence * 100).toFixed(1);
    confidenceValue.textContent = confidence + '%';

    // Rotate gauge (0% is -45deg, 100% is 135deg)
    const rotation = -45 + (data.confidence * 180);
    gaugeFill.style.transform = `rotate(${rotation}deg)`;

    // Update Explanation Highlights
    if (data.words && data.weights) {
        textHighlighted.innerHTML = data.words.map((word, i) => {
            const weight = data.weights[i];
            const opacity = Math.min(weight * 10, 1); // Scale for visibility
            const color = data.sentiment === 'Positive' ? '34, 197, 94' : '239, 68, 68';
            return `<span class="highlight" style="background: rgba(${color}, ${opacity})">${word}</span>`;
        }).join(' ');
    }
}
