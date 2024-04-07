// Function to speak the welcome message and start voice recognition
function speakWelcomeMessage() {
    var welcomeMsg = new SpeechSynthesisUtterance();
    welcomeMsg.text = "Welcome. To our services rightnow we have three services First one is Book your appointment and second is get your heart report ananlysis and last one is Find Hospital nearby me Please say which service u want to use.";
    window.speechSynthesis.speak(welcomeMsg);

    var recognition = new webkitSpeechRecognition();
    recognition.lang = "en-US";
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.onresult = function(event) {
        var result = event.results[0][0].transcript.toLowerCase();
        if (result.includes("appointment booking")) {
            window.location.href = "#";
        }
        else if(result.includes("Heart Report Analysis")) {
            window.location.href = "report.html";
        }
        else if(result.includes("Heart Hospital Nearby me")) {
            window.location.href = "#";
        } else {
            var errorMsg = new SpeechSynthesisUtterance();
            errorMsg.text = "Sorry, I didn't understand. Please say 'which service u want' to proceed.";
            window.speechSynthesis.speak(errorMsg);
        }
    };
    recognition.onerror = function(event) {
        alert("Voice recognition error occurred. Please try again.");
    };
    recognition.start();
}

// Speak the welcome message and start voice recognition when the page loads
speakWelcomeMessage();
