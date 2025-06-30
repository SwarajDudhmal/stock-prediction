import { initializeApp } from "https://www.gstatic.com/firebasejs/11.2.0/firebase-app.js";
import { getAuth, GoogleAuthProvider, signInWithPopup, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/11.2.0/firebase-auth.js";

// Firebase Configuration
const firebaseConfig = {
    apiKey: "AIzaSyBKzklTetliRIv9UunBkZ-JfqV5mQnf3PU",
    authDomain: "login-form-784b8.firebaseapp.com",
    projectId: "login-form-784b8",
    storageBucket: "login-form-784b8.firebasestorage.app",
    messagingSenderId: "444698526985",
    appId: "1:444698526985:web:c032a38d5cd43af9cbbfd8"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth();
const provider = new GoogleAuthProvider();

// Function to show messages
function showMessage(message, divId) {
    const messageDiv = document.getElementById(divId);
    messageDiv.style.display = "block";
    messageDiv.innerHTML = message;
    messageDiv.style.opacity = 1;
    setTimeout(() => {
        messageDiv.style.opacity = 0;
    }, 5000);
}

// Check if the user is already signed in
onAuthStateChanged(auth, (user) => {
    if (user) {
        // If the user is already signed in, redirect them to the homepage
        window.location.href = 'html_files/homepage1.html'; // Redirect to homepage1.html after successful login
    }
});

// Google Sign-Up
document.getElementById('googleSignUp').addEventListener('click', () => {
    signInWithPopup(auth, provider)
        .then((result) => {
            const user = result.user;
            showMessage('Signed up successfully with Google!', 'signUpMessage');
            console.log(user);  // Log user details to the console
            // Redirect to the homepage
            window.location.href = 'html_files/homepage1.html'; // Redirect after successful Google sign-up
        })
        .catch((error) => {
            console.error('Error during Google Sign-Up:', error);
            showMessage('Google Sign-Up failed.', 'signUpMessage');
        });
});

// Google Sign-In
document.getElementById('googleSignIn').addEventListener('click', () => {
    signInWithPopup(auth, provider)
        .then((result) => {
            const user = result.user;
            showMessage('Logged in successfully with Google!', 'signInMessage');
            console.log(user);  // Log user details to the console
            // Redirect to the homepage
            window.location.href = 'html_files/homepage1.html'; // Redirect after successful Google sign-in
        })
        .catch((error) => {
            console.error('Error during Google Sign-In:', error);
            showMessage('Google Sign-In failed.', 'signInMessage');
        });
});
