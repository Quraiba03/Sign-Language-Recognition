const videoFeed = document.getElementById("videoFeed");
const maximizeBtn = document.getElementById("maximizeBtn");

maximizeBtn.addEventListener("click", () => {
    videoFeed.classList.toggle("maximized");
    if (videoFeed.classList.contains("maximized")) {
        maximizeBtn.innerText = "Restore";
    } else {
        maximizeBtn.innerText = "Maximize";
    }
});
