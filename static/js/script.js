document.getElementById("prediction-form").addEventListener("submit", function(event) {
    event.preventDefault();
    
    const X1 = parseFloat(document.getElementById("X1").value);
    const X2_2 = document.getElementById("X2_2").checked ? 1 : 0;
    const X3_2 = (  document.getElementById("X3_2").value);
    const X4_2 = (  document.getElementById("X4_2").value);
    const X5 = parseFloat(document.getElementById("X5").value);
    const X6 = parseFloat(document.getElementById("X6").value);
    const X7 = parseFloat(document.getElementById("X7").value);
    const X8 = parseFloat(document.getElementById("X8").value);
    const X9 = parseFloat(document.getElementById("X9").value);
    const X10 = parseFloat(document.getElementById("X10").value);
    const X11 = parseFloat(document.getElementById("X11").value);
    const X12 = parseFloat(document.getElementById("X12").value);
    const X13 = parseFloat(document.getElementById("X13").value);
    const X14 = parseFloat(document.getElementById("X14").value);
    const X15 = parseFloat(document.getElementById("X15").value);
    const X16 = parseFloat(document.getElementById("X16").value);
    const X17 = parseFloat(document.getElementById("X17").value);
    const X18 = parseFloat(document.getElementById("X18").value);
    const X19 = parseFloat(document.getElementById("X19").value);
    const X20 = parseFloat(document.getElementById("X20").value);
    const X21 = parseFloat(document.getElementById("X21").value);
    const X22 = parseFloat(document.getElementById("X22").value);
    const X23 = parseFloat(document.getElementById("X23").value);
    const X24 = parseFloat(document.getElementById("X24").value);

    fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            X1, X2_2, X3_2, X4_2, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14,
            X15, X16, X17, X18, X19, X20, X21, X22, X23, X24
        })
    })
    .then(response => response.json())
    .then(data => {
        const predictionText = data.prediction === 1 ? "Inadimplente" : "Adimplente";
        document.getElementById("prediction").innerText = "Classe: " + predictionText;
        document.getElementById("result").style.display = "block";
    })
    .catch(error => {
        console.error('Erro ao fazer a previsão:', error);
    });
});