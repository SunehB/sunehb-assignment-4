document.getElementById('search-form').addEventListener('submit', function (event) {
    event.preventDefault();
    
    let query = document.getElementById('query').value;
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            'query': query
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        displayResults(data);
        displayChart(data);
    });
});

function displayResults(data) {
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<h2>Results</h2>';
    for (let i = 0; i < data.documents.length; i++) {
        let docDiv = document.createElement('div');
        docDiv.innerHTML = `<strong>Document ${data.indices[i]}</strong><p>${data.documents[i]}</p><br><strong>Similarity: ${data.similarities[i]}</strong>`;
        resultsDiv.appendChild(docDiv);
    }
}

function displayChart(data) {
    // Input: data (object) - contains the following keys:
    //        - documents (list) - list of documents
    //        - indices (list) - list of indices   
    //        - similarities (list) - list of similarities
    // TODO: Implement function to display chart here
    //       There is a canvas element in the HTML file with the id 'similarity-chart'

    const ctx = document.getElementById('similarity-chart').getContext('2d');

    //delete the previous chart
    if (window.similarityChart) {
        window.similarityChart.destroy();
    }

    window.similarityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.indices.map(i => `Doc ${i + 1}`), // Labels for each bar
            datasets: [{
                label: 'Cosine Similarity',
                data: data.similarities, // Similarity scores
                backgroundColor: 'rgba(75, 192, 192, 0.6)', // Bar color
                borderColor: 'rgba(75, 192, 192, 1)', // Border color
                borderWidth: 1 // Border width
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true // Start the y-axis at zero
                }
            },
            responsive: true, // Make the chart responsive
            maintainAspectRatio: false // Do not maintain the aspect ratio
        }
    });




}