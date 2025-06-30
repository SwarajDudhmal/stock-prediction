import { initializeApp } from "https://www.gstatic.com/firebasejs/11.2.0/firebase-app.js";
import { getAuth, sendPasswordResetEmail } from "https://www.gstatic.com/firebasejs/11.2.0/firebase-auth.js";

// Your web app's Firebase configuration
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

function showMessage(message, divId) {
    const messageDiv = document.getElementById(divId);
    messageDiv.style.display = "block";
    messageDiv.innerHTML = message;
    messageDiv.style.opacity = 1;
    setTimeout(() => {
        messageDiv.style.opacity = 0;
    }, 5000);
}

const forgotPasswordForm = document.getElementById('forgot-password-form');
forgotPasswordForm.addEventListener('submit', (event) => {
    event.preventDefault();

    const email = document.getElementById('reset-email').value;
    const auth = getAuth();

    // Send password reset email
    sendPasswordResetEmail(auth, email)
        .then(() => {
            showMessage('Password reset email sent successfully. Please check your inbox.', 'message');
            // Optionally, redirect to login page after a delay
            setTimeout(() => {
                window.location.href = '../index.html'; // Redirect to the SignIn page
            }, 3000);
        })
        .catch((error) => {
            console.error("Error sending password reset email:", error.message);
            showMessage('Error: Unable to send reset email. Please try again later.', 'message');
        });
});
