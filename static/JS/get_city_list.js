fetch('city_list')
    .then(response => response.json())
    .then(data => {
        data.forEach(city => {
            let newElement = document.createElement("p");
            newElement.textContent = city;
            newElement.addEventListener('click', function() {
                fetch('/process_city', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ city: city })
                })
                .then(response => response.json())
                .then(data => {
                    console.log(data.message);
                    console.log(data.plot);

                    // Create a new URL object for the plotting page
                    const plottingUrl = new URL("http://127.0.0.1:5000/plotting");

                    // Set city and img_str as query parameters
                    plottingUrl.searchParams.set('city', city);
                    plottingUrl.searchParams.set('img_str', data.plot);

                    // Redirect to the plotting page
                    window.location.href = plottingUrl.toString();
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            });
            document.getElementById('city_container').appendChild(newElement)
        });
    });
