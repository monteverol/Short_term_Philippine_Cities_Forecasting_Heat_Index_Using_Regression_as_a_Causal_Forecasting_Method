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
                    console.log(data.predictions);

                    // Create a new URL object for the plotting page
                    const dateUrl = new URL("http://127.0.0.1:5000/select_date");

                    // Set city and img_str as query parameters
                    dateUrl.searchParams.set('city', city);
                    dateUrl.searchParams.set('predictions', data.predictions);

                    // Redirect to the plotting page
                    window.location.href = dateUrl.toString();
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            });
            document.getElementById('city_container').appendChild(newElement)
        });
    });
