() =>{
  if (window.location.hostname.endsWith(".hf.space")) {
    const hfHeader = document.getElementById("huggingface-space-header");
    if (hfHeader) {
      hfHeader.style.display = "none";
    }
  }
}
