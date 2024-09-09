() =>{
  const menu = document.querySelector("#menu");
  const menuButton = menu.querySelector("button");
  const content = document.querySelector("#content");

  const updateMenu = () => {
    const isOpen = menuButton.classList.contains("open");
    content.style.display = isOpen ? "none" : "flex";
  };

  const observer = new MutationObserver(updateMenu);
  observer.observe(menuButton, { attributes: true, attributeFilter: ["class"] });
  updateMenu();

  if (window.location.hostname.endsWith(".hf.space")) {
    const hfHeader = document.getElementById("huggingface-space-header");
    if (hfHeader) {
      hfHeader.style.display = "none";
    }
  }
}
