// Select the cursor line element
const cursorLine = document.getElementById('cursor-line');

// Listen for mouse movement to track the cursor's position
document.addEventListener('mousemove', (e) => {
    // Get the cursor's X and Y position
    const mouseX = e.clientX;
    const mouseY = e.clientY;
    
    // Position the cursor line
    cursorLine.style.left = `${mouseX - cursorLine.offsetWidth / 2}px`;
    cursorLine.style.top = `${mouseY - cursorLine.offsetHeight / 2}px`;
    
    // Make the cursor line visible
    cursorLine.style.opacity = 1;
    
    // Hide the line after 1 second
    setTimeout(() => {
        cursorLine.style.opacity = 0;
    }, 1000); // Line disappears after 1 second
});
