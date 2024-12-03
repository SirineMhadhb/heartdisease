const express = require('express');
const bodyParser = require('body-parser');
const app = express();


app.use(bodyParser.json());


let users = [];


app.post('/users', (req, res) => {
    const { username, email } = req.body;

   
    if (!username || !email) {
        return res.status(400).json({ error: 'Missing required data: username and email are required.' });
    }

    const newUser = {
        id: users.length + 1,
        username,
        email
    };

    
    users.push(newUser);

   
    return res.status(201).json(newUser);
});


const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});

module.exports = app; 