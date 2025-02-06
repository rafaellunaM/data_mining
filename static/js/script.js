document.getElementById("prediction-form").addEventListener("submit", function(event) {
    event.preventDefault();
    const sepalLength = parseFloat(document.getElementById("sepal_length").value);
    const sepalWidth = parseFloat(document.getElementById("sepal_width").value);
    const petalLength = parseFloat(document.getElementById("petal_length").value);
    const petalWidth = parseFloat(document.getElementById("petal_width").value);

    fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            sepal_length: sepalLength,
            sepal_width: sepalWidth,
            petal_length: petalLength,
            petal_width: petalWidth
        })
    })
    .then(response => response.json())
    .then(data => {
        // Exibir o resultado
        document.getElementById("prediction").innerText = "Classe: " + data.prediction;
        document.getElementById("class_name").innerText = "Nome da classe: " + data.class_name;
        document.getElementById("result").style.display = "block";
    })
    .catch(error => {
        console.error('Erro ao fazer a previsão:', error);
    });
});
