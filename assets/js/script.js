(() => {
  const menuToggle = document.getElementById("menu-toggle");
  const menu = document.getElementById("menu");

  if (menuToggle && menu) {
    menuToggle.addEventListener("click", () => {
      const expanded = menuToggle.getAttribute("aria-expanded") === "true";
      menuToggle.setAttribute("aria-expanded", String(!expanded));
      menu.classList.toggle("open");
    });
  }

  const heroNodes = document.getElementById("heroNodes");
  if (heroNodes) {
    for (let i = 0; i < 20; i++) {
      const node = document.createElement("div");
      node.className = "node";
      node.style.left = `${Math.random() * 100}%`;
      node.style.animationDuration = `${8 + Math.random() * 12}s`;
      node.style.animationDelay = `${Math.random() * 10}s`;
      const size = 3 + Math.random() * 5;
      node.style.width = `${size}px`;
      node.style.height = `${size}px`;
      heroNodes.appendChild(node);
    }
  }

  // Active le cache navigateur via service worker.
  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
      navigator.serviceWorker.register("/service-worker.js").catch(() => {
        // Ignore silencieusement l'erreur d'enregistrement.
      });
    });
  }
})();
