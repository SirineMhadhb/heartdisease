// Wrap the entire content in an immediately-invoked asynchronous function
(async () => {
    const chai = await import('chai');
    const assert = chai.assert;
    const axios = require('axios');
    const app = require('./app'); // Import your Express application

    describe('API Tests', function() {
        it('Should create a new user', async function() {
            // Send a POST request to create a new user
            const response = await axios.post('http://localhost:3000/users', {
                username: 'john_doe',
                email: 'john@example.com'
            });

            // Check if the response is successful
            assert.equal(response.status, 201);

            // Check the response to ensure it contains the ID of the created user
            assert.exists(response.data.id);
        });
    });

    // Run the Mocha test suite
    run();
})();
