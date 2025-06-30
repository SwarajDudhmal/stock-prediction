// Import Firebase SDK
import { 
  initializeApp 
} from "https://www.gstatic.com/firebasejs/11.2.0/firebase-app.js";
import { 
  getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword, 
  sendEmailVerification, signOut, GoogleAuthProvider, signInWithPopup 
} from "https://www.gstatic.com/firebasejs/11.2.0/firebase-auth.js";
import { 
  getFirestore, setDoc, doc, getDoc 
} from "https://www.gstatic.com/firebasejs/11.2.0/firebase-firestore.js";

// Firebase Configuration
const firebaseConfig = {
  apiKey: "AIzaSyBKzklTetliRIv9UunBkZ-JfqV5mQnf3PU",
  authDomain: "login-form-784b8.firebaseapp.com",
  projectId: "login-form-784b8",
  storageBucket: "login-form-784b8.appspot.com",
  messagingSenderId: "444698526985",
  appId: "1:444698526985:web:c032a38d5cd43af9cbbfd8"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth();
const db = getFirestore();
const provider = new GoogleAuthProvider();

// Function to show messages for 3 to 5 seconds
function showMessage(message, divId) {
  const messageDiv = document.getElementById(divId);
  messageDiv.style.display = "block";
  messageDiv.innerHTML = message;
  messageDiv.style.opacity = 1;
  setTimeout(() => {
    messageDiv.style.opacity = 0;
  }, 1000); // Show message for 4 seconds (between 3 to 5 sec)
}

// 🛑 SIGN-UP FUNCTION WITH EMAIL VERIFICATION & AUTO REDIRECT
document.getElementById('submitSignUp').addEventListener('click', (event) => {
  event.preventDefault();

  const email = document.getElementById('rEmail').value;
  const password = document.getElementById('rPassword').value;
  const firstName = document.getElementById('fName').value;
  const lastName = document.getElementById('lName').value;

  createUserWithEmailAndPassword(auth, email, password)
    .then((userCredential) => {
      const user = userCredential.user;

      sendEmailVerification(user)
        .then(() => {
          showMessage('Verification email sent! Please check your inbox.', 'signUpMessage');
          
          // Save user details in Firestore
          const userData = { email, firstName, lastName };
          const docRef = doc(db, "users", user.uid);
          
          setDoc(docRef, userData)
            .then(() => {
              signOut(auth).then(() => {
                showMessage('Please verify your email before logging in.', 'signUpMessage');
                setTimeout(() => {
                  window.location.href = 'index.html'; // Redirect to sign-in page after message disappears
                }, 1000);
              });
            })
            .catch((error) => {
              console.error("Error writing document", error);
            });
        })
        .catch((error) => {
          console.error("Error sending email verification", error);
          showMessage('Failed to send verification email.', 'signUpMessage');
        });
    })
    .catch((error) => {
      if (error.code === 'auth/email-already-in-use') {
        showMessage('Email already in use!', 'signUpMessage');
      } else {
        showMessage('Signup failed!', 'signUpMessage');
      }
    });
});

// ✅ SIGN-IN FUNCTION WITH EMAIL VERIFICATION CHECK
document.getElementById('submitSignIn').addEventListener('click', (event) => {
  event.preventDefault();
  
  const email = document.getElementById('email').value;
  const password = document.getElementById('password').value;

  signInWithEmailAndPassword(auth, email, password)
    .then((userCredential) => {
      const user = userCredential.user;

      if (user.emailVerified) {
        showMessage('Login successful!', 'signInMessage');
        setTimeout(() => {
          localStorage.setItem('loggedInUserId', user.uid);
          window.location.href = 'html_files/homepage1.html';
        }, 1000);
      } else {
        showMessage('Please verify your email before logging in.', 'signInMessage');
        setTimeout(() => {
          signOut(auth).then(() => {
            window.location.href = 'index.html';
          });
        }, 1000); // Show error for 4 seconds before redirecting
      }
    })
    .catch((error) => {
      if (error.code === 'auth/invalid-credential') {
        showMessage('Incorrect email or password.', 'signInMessage');
      } else {
        showMessage('Account does not exist.', 'signInMessage');
      }
    });
});

// 🔹 GOOGLE SIGN-IN FUNCTION
document.getElementById('googleSignIn').addEventListener('click', () => {
  signInWithPopup(auth, provider)
    .then((result) => {
      const user = result.user;
      const docRef = doc(db, "users", user.uid);
      
      getDoc(docRef).then((docSnap) => {
        if (!docSnap.exists()) {
          setDoc(docRef, { email: user.email, name: user.displayName })
            .then(() => console.log("User data saved"));
        }
      });
      
      showMessage('Google Sign-In Successful!', 'signInMessage');
      setTimeout(() => {
        window.location.href = 'html_files/homepage1.html';
      }, 1000);
    })
    .catch((error) => {
      console.error("Google Sign-In Error", error);
      showMessage('Google Sign-In failed!', 'signInMessage');
    });
});
