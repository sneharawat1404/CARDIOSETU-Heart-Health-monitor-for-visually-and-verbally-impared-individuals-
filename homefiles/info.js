const loginText = document.querySelector(".title-text .login");
const loginForm = document.querySelector("form.login");
const loginBtn = document.querySelector("label.login");
const signupBtn = document.querySelector("label.signup");
const signupLink = document.querySelector("form .signup-link a");
signupBtn.onclick = (()=>{
  loginForm.style.marginLeft = "-50%";
  loginText.style.marginLeft = "-50%";
});
loginBtn.onclick = (()=>{
  loginForm.style.marginLeft = "0%";
  loginText.style.marginLeft = "0%";
});
signupLink.onclick = (()=>{
  signupBtn.click();
  return false;
});

if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
    console.log('SpeechRecognition API is supported.');
} else {
    console.log('SpeechRecognition API is not supported.');
}

if ('speechSynthesis' in window) {
    console.log('SpeechSynthesis API is supported.');
} else {
    console.log('SpeechSynthesis API is not supported.');
}
console.log()
