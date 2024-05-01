fetch('city_list')
    .then(response => response.json())
    .then(data => {
        data.forEach(city => {
            const url = new URL("http://127.0.0.1:5000/plotting");
            url.searchParams.set('city', city);
            let newElement = `
            <p onclick="window.location.replace('${url.toString()}')">
                ${city}
            </p>
            `;
            document.getElementById('city_container').innerHTML += newElement;
        });
    });
