/**
 * Stoolap website JavaScript
 */

document.addEventListener('DOMContentLoaded', () => {
  // Remove preload class immediately
  document.body.classList.remove('preload');

  // Initialize theme toggle
  initThemeToggle();
  
  // Initialize mobile menu
  initMobileMenu();
  
  // Initialize sidebar navigation
  initSidebarNav();
  
  // Add copy buttons to code blocks
  addCodeCopyButtons();
  
  // Initialize SVG theme sync
  initSvgThemeSync();
});

/**
 * Initialize theme toggle functionality
 */
function initThemeToggle() {
  const themeToggle = document.getElementById('theme-toggle');
  if (!themeToggle) return;
  
  // Get current theme from data-theme attribute
  const getCurrentTheme = () => document.documentElement.getAttribute('data-theme') || 'light';
  
  // Toggle between light and dark themes
  themeToggle.addEventListener('click', () => {
    const currentTheme = getCurrentTheme();
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    // Update HTML attribute
    document.documentElement.setAttribute('data-theme', newTheme);
    
    // Save preference to localStorage
    localStorage.setItem('theme', newTheme);
  });
}

/**
 * Initialize mobile menu toggle
 */
function initMobileMenu() {
  const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
  const mainNav = document.querySelector('.main-nav');
  
  if (!mobileMenuToggle || !mainNav) return;
  
  mobileMenuToggle.addEventListener('click', () => {
    mainNav.classList.toggle('active');
    
    // Update aria-expanded state
    const isExpanded = mainNav.classList.contains('active');
    mobileMenuToggle.setAttribute('aria-expanded', isExpanded);
    mobileMenuToggle.classList.toggle('active', isExpanded);
    
    // Prevent scrolling when menu is open
    document.body.style.overflow = isExpanded ? 'hidden' : '';
  });
  
  // Close mobile menu when clicking outside
  document.addEventListener('click', (event) => {
    if (
      mainNav && 
      mainNav.classList.contains('active') && 
      !mainNav.contains(event.target) && 
      !mobileMenuToggle.contains(event.target)
    ) {
      mainNav.classList.remove('active');
      mobileMenuToggle.setAttribute('aria-expanded', 'false');
      mobileMenuToggle.classList.remove('active');
      document.body.style.overflow = '';
    }
  });
  
  // Handle window resize
  window.addEventListener('resize', () => {
    if (window.innerWidth > 768 && mainNav.classList.contains('active')) {
      mainNav.classList.remove('active');
      mobileMenuToggle.setAttribute('aria-expanded', 'false');
      mobileMenuToggle.classList.remove('active');
      document.body.style.overflow = '';
    }
  });
}

/**
 * Initialize sidebar navigation toggles
 */
function initSidebarNav() {
  const navToggles = document.querySelectorAll('.docs-nav-toggle');
  
  navToggles.forEach(toggle => {
    toggle.addEventListener('click', () => {
      const isExpanded = toggle.getAttribute('aria-expanded') === 'true';
      toggle.setAttribute('aria-expanded', !isExpanded);
    });
  });
  
  // On mobile, collapse all sections initially except the active one
  if (window.innerWidth <= 768) {
    const currentPageLink = document.querySelector('.docs-nav a[aria-current="page"]');
    
    if (currentPageLink) {
      const currentSection = currentPageLink.closest('.docs-nav-section');
      
      navToggles.forEach(toggle => {
        const isCurrentSection = toggle.parentElement === currentSection;
        toggle.setAttribute('aria-expanded', isCurrentSection);
      });
    }
  }
}

/**
 * Add copy buttons to code blocks
 */
function addCodeCopyButtons() {
  const codeBlocks = document.querySelectorAll('pre');
  
  codeBlocks.forEach(block => {
    // Only add button if not already present
    if (block.querySelector('.copy-button')) return;
    
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.type = 'button';
    button.ariaLabel = 'Copy code to clipboard';
    button.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16">
        <path d="M6.9998 6V3C6.9998 2.44772 7.44752 2 7.9998 2H19.9998C20.5521 2 20.9998 2.44772 20.9998 3V17C20.9998 17.5523 20.5521 18 19.9998 18H16.9998V20.9991C16.9998 21.5519 16.5499 22 15.993 22H4.00666C3.45059 22 3 21.5554 3 20.9991L3.0026 7.00087C3.0027 6.44811 3.45264 6 4.00942 6H6.9998ZM5.00242 8L5.00019 20H14.9998V8H5.00242ZM8.9998 6H16.9998V16H18.9998V4H8.9998V6Z" />
      </svg>
    `;
    
    const codeElement = block.querySelector('code');
    const code = codeElement ? codeElement.innerText : block.innerText;
    
    button.addEventListener('click', () => {
      navigator.clipboard.writeText(code).then(() => {
        button.innerHTML = `
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16">
            <path d="M9.9997 15.1709L19.1921 5.97852L20.6063 7.39273L9.9997 17.9993L3.63574 11.6354L5.04996 10.2212L9.9997 15.1709Z" />
          </svg>
        `;
        
        setTimeout(() => {
          button.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16">
              <path d="M6.9998 6V3C6.9998 2.44772 7.44752 2 7.9998 2H19.9998C20.5521 2 20.9998 2.44772 20.9998 3V17C20.9998 17.5523 20.5521 18 19.9998 18H16.9998V20.9991C16.9998 21.5519 16.5499 22 15.993 22H4.00666C3.45059 22 3 21.5554 3 20.9991L3.0026 7.00087C3.0027 6.44811 3.45264 6 4.00942 6H6.9998ZM5.00242 8L5.00019 20H14.9998V8H5.00242ZM8.9998 6H16.9998V16H18.9998V4H8.9998V6Z" />
            </svg>
          `;
        }, 2000);
      }).catch(err => {
        console.error('Could not copy text: ', err);
      });
    });
    
    block.style.position = 'relative';
    block.appendChild(button);
    
    // Add copy button styling
    const style = document.createElement('style');
    style.textContent = `
      pre {
        position: relative;
      }
      
      .copy-button {
        position: absolute;
        top: 8px;
        right: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        background-color: rgba(var(--color-bg-rgb, 255, 255, 255), 0.1);
        border: 1px solid rgba(var(--color-border-rgb, 200, 200, 200), 0.4);
        border-radius: 4px;
        opacity: 0;
        transition: opacity 0.2s, background-color 0.2s;
        cursor: pointer;
        color: var(--color-text-muted, #666);
      }
      
      pre:hover .copy-button {
        opacity: 1;
      }
      
      .copy-button:hover {
        background-color: rgba(var(--color-bg-rgb, 255, 255, 255), 0.2);
        color: var(--color-primary, #0f7d3c);
      }
      
      .copy-button:active {
        background-color: rgba(var(--color-bg-rgb, 255, 255, 255), 0.3);
      }
      
      .copy-button svg {
        fill: currentColor;
      }
    `;
    
    document.head.appendChild(style);
  });
}

/**
 * Set up table of contents for documentation pages
 */
function setupTableOfContents() {
  const docContent = document.querySelector('.doc-content');
  if (!docContent) return;
  
  const headings = docContent.querySelectorAll('h2, h3');
  if (headings.length < 3) return; // Only show TOC if we have enough headings
  
  const toc = document.createElement('nav');
  toc.className = 'table-of-contents';
  toc.setAttribute('aria-label', 'Table of Contents');
  
  const tocTitle = document.createElement('h2');
  tocTitle.textContent = 'On this page';
  tocTitle.className = 'toc-title';
  
  const tocList = document.createElement('ul');
  
  headings.forEach((heading, index) => {
    // Add ID to heading if it doesn't have one
    if (!heading.id) {
      heading.id = `heading-${index}`;
    }
    
    const listItem = document.createElement('li');
    listItem.className = `toc-${heading.tagName.toLowerCase()}`;
    
    const link = document.createElement('a');
    link.href = `#${heading.id}`;
    link.textContent = heading.textContent;
    
    listItem.appendChild(link);
    tocList.appendChild(listItem);
  });
  
  toc.appendChild(tocTitle);
  toc.appendChild(tocList);
  
  // Insert TOC after doc header
  const docHeader = document.querySelector('.doc-header');
  if (docHeader) {
    docHeader.insertAdjacentElement('afterend', toc);
  } else {
    docContent.insertAdjacentElement('beforebegin', toc);
  }
  
  // Add TOC styles
  const tocStyle = document.createElement('style');
  tocStyle.textContent = `
    .table-of-contents {
      margin-bottom: 2rem;
      padding: 1.5rem;
      background-color: var(--color-bg-alt);
      border-radius: 0.5rem;
      border: 1px solid var(--color-border);
    }
    
    .toc-title {
      margin-top: 0;
      margin-bottom: 1rem;
      font-size: 1.2rem;
      font-weight: 600;
      border-bottom: none;
      padding-bottom: 0;
    }
    
    .table-of-contents ul {
      margin-bottom: 0;
    }
    
    .table-of-contents li {
      margin-bottom: 0.25rem;
    }
    
    .toc-h3 {
      margin-left: 1.5rem;
      font-size: 0.9rem;
    }
    
    .table-of-contents a {
      color: var(--color-text-muted);
      text-decoration: none;
    }
    
    .table-of-contents a:hover {
      color: var(--color-primary);
      text-decoration: underline;
    }
  `;
  
  document.head.appendChild(tocStyle);
}

// Initialize table of contents for doc pages
if (document.querySelector('.doc-article')) {
  setupTableOfContents();
}

/**
 * Sync SVGs with the current theme
 */
function initSvgThemeSync() {
  // Initial sync on load
  syncSvgThemes();
  
  // Watch for theme changes
  const observer = new MutationObserver(mutations => {
    mutations.forEach(mutation => {
      if (mutation.attributeName === 'data-theme') {
        syncSvgThemes();
      }
    });
  });
  
  // Start observing theme changes on the HTML element
  observer.observe(document.documentElement, { attributes: true });
}

/**
 * Update all SVG objects to match the current theme
 */
function syncSvgThemes() {
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
  
  // Find all SVG objects
  const svgObjects = document.querySelectorAll('object[type="image/svg+xml"]');
  
  svgObjects.forEach(obj => {
    obj.addEventListener('load', function() {
      try {
        // Access the SVG document inside the object
        const svgDoc = obj.contentDocument;
        if (svgDoc && svgDoc.documentElement) {
          // Set data-theme attribute on the SVG root element
          svgDoc.documentElement.setAttribute('data-theme', currentTheme);
        }
      } catch (e) {
        console.error('Error updating SVG theme:', e);
      }
    });
    
    // Reload already loaded SVGs to trigger the load event
    if (obj.contentDocument && obj.contentDocument.readyState === 'complete') {
      const currentSrc = obj.getAttribute('data');
      obj.setAttribute('data', '');
      obj.setAttribute('data', currentSrc);
    }
  });
}