const firebaseConfig = {
  //   copy your firebase config informations
  apiKey: "AIzaSyDuCaYTMHm7xFcOp163p9oXjNeYjs8wE5M",
  authDomain: "demo1-746fb.firebaseapp.com",
  databaseURL: "https://demo1-746fb-default-rtdb.firebaseio.com",
  projectId: "demo1-746fb",
  storageBucket: "demo1-746fb.firebasestorage.app",
  messagingSenderId: "222913561439",
  appId: "1:222913561439:web:7f8f0b3589113f01efacea"
};

// initialize firebase
firebase.initializeApp(firebaseConfig);

// reference your database
var contactFormDB = firebase.database().ref("contactForm");

document.getElementById("contactForm").addEventListener("submit", submitForm);

function submitForm(e) {
  e.preventDefault();

  var name = getElementVal("name");
  var emailid = getElementVal("emailid");
  var msgContent = getElementVal("msgContent");

  saveMessages(name, emailid, msgContent);

  //   enable alert
  document.querySelector(".alert").style.display = "block";

  //   remove the alert
  setTimeout(() => {
    document.querySelector(".alert").style.display = "none";
  }, 3000);

  //   reset the form
  document.getElementById("contactForm").reset();
}

const saveMessages = (name, emailid, msgContent) => {
  var newContactForm = contactFormDB.push();

  newContactForm.set({
    name: name,
    emailid: emailid,
    msgContent: msgContent,
  });
};

const getElementVal = (id) => {
  return document.getElementById(id).value;
};
