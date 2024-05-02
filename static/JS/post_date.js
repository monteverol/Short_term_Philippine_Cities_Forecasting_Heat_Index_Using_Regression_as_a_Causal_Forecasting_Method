const day = document.getElementById("day")
const month = document.getElementById("month")
const year = document.getElementById("year")

document.getElementById("selectDateNext").addEventListener('click', () => {
    fetch('/process_select_date', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ date: `${year.value}-${month.value}-${day.value}` })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message)
        console.log(data.date)
    })
});