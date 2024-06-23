modules = [
    {
        name: "Regression and Classification",
        filename: "01_Regression.md",
    },
    {
        name: "Popular ML Models",
        filename: "02_Popular_ML_Models.md",
    },
    {
        name: "Clustering",
        filename: "03_Clustering.md",
    },
    {
        name: "Ensemble Methods",
        filename: "04_Ensemble_Methods.md",
    },
    {
        name: "Neural Networks",
        filename: "05_Neural_Networks.md",
    },
    {
        name: "Image Processing",
        filename: "06_Image_Processing.md",
    },
    {
        name: "Convolutional Neural Networks",
        filename: "07_Convolutional_Neural_Networks.md",
    },
    {
        name: "Transfer Learning and Autoencoders",
        filename: "08_Transfer_Learning_and_Autoencoders.md",
    },
    {
        name: "Natural Language Processing",
        filename: "09_Natural_Language_Processing.md",
    },
    {
        name: "Recurrent Neural Networks",
        filename: "10_Recurrent_Neural_Networks.md",
    },
    {
        name: "Transformers",
        filename: "11_Transformers.md",
    },
    {
        name: "Language Models and LLMs",
        filename: "12_Language_Models_and_LLMs.md",
    },
    {
        name: "Foundational Models",
        filename: "13_Foundational_Models.md",
    },
];

projects = [
    {
        name: "Flower classification with NN",
        filename: "flower_classification.md",
    },
];
papers = [
    {
        name: "Attention is all you need",
        filename: "attention_is_all_you_need.md",
    },
];

// Get the URL search parameters
const searchParams = new URLSearchParams(window.location.search);

// Get the id parameter value and select array, folder and  module accordingly.
let type = searchParams.get("type");

if (type == "project") {
    arr = projects;
    folder = "Projects";
} else if (type == "paper") {
    arr = papers;
    folder = "Papers";
} else {
    type = 'module'
    arr = modules;
    folder = "Modules";
}

let id = parseInt(searchParams.get("id"));

if (!(id > 1 && id <= arr.length)) {
    id = 1;
}

const module = arr[parseInt(id) - 1];


// Function that takes HTML of pre block and returns it with copy code button
function addCopyButtonToPre(preInnerHTML) {

    // Create heading div
    btn = `<button class = 'copy-btn'><i class="fa-regular fa-copy"></i>  Copy Code</button>`;

    // Append the Copy button to the code block
    preInnerHTML += btn;

    // Return the modified inner HTML of the temporary <div>
    return preInnerHTML;
}

// Function to fetch the README file
async function fetchReadme() {
    try {
        const response = await fetch(`../${folder}/${module.filename}`); // Ensure your README file is accessible
        if (!response.ok) {
            throw new Error("Network response was not ok");
        }
        const markdownText = await response.text();
        return markdownText;
    } catch (error) {
        return `<h1>Contents coming soon...</h1>`;
    }
}

// Function to convert markdown to HTML and display it
async function displayReadme() {
    const markdownText = await fetchReadme();
    if (markdownText) {
        const converter = new showdown.Converter();
        const htmlContent = converter.makeHtml(markdownText);
        document.querySelector(".content").innerHTML =
            htmlContent + generatePrevNextBtns();
    }
}

// Function that generates next and prev page button according to id
function generatePrevNextBtns() {
    const nextId = id + 1;
    const prevId = id - 1;
    btns = document.createElement('div');
    btns.innerHTML = `
    <div class="module-change-btns">
    <a href="?type=${type}&id=${prevId}" class="prev-btn"><i class="fa-solid fa-chevron-left"></i> PREVIOUS</a>
    <a href="?type=${type}&id=${nextId}" class="next-btn">NEXT <i class="fa-solid fa-chevron-right"></i></a>
    </div>`;

    if (id <= 1) {
        btns.querySelector('.prev-btn').classList.add('disabled')
    } if (id >= arr.length) {
        btns.querySelector('.next-btn').classList.add('disabled')

    }

    return btns.innerHTML;
}


// Perform other JS code after readme file is loaded
async function main() {
    await displayReadme();

    // Display Module name
    document.querySelector(
        ".title"
    ).innerHTML = `Module ${id}: ${module.name} `;
    allChapters = [];

    // Display all chapters
    chapterList = document.querySelector(".chapter-list");
    chapterList.innerHTML = "";

    document.querySelectorAll(".content h1").forEach((header) => {
        chapterList.innerHTML += `<li><a href="#${header.id}">${header.innerHTML}</a></li>`;
    });

    preBlocks = document.querySelectorAll("pre");
    preBlocks.forEach((pre) => {
        if(!pre.querySelector('code').classList.contains('language-output')){
            
            pre.innerHTML = addCopyButtonToPre(pre.innerHTML);
            pre.querySelector(".copy-btn").onclick = function () {
                // Select the code inside the code block
                const range = document.createRange();
                range.selectNode(pre.querySelector("code"));
                window.getSelection().removeAllRanges();
                window.getSelection().addRange(range);
                document.execCommand("copy");
                window.getSelection().removeAllRanges();
                pre.querySelector(".copy-btn").innerHTML =
                    '<i class="fa-regular fa-copy"></i> Code Copied!';
                setTimeout(() => {
                    pre.querySelector(".copy-btn").innerHTML =
                        '<i class="fa-regular fa-copy"></i> Copy Code';
                }, 2000);
            };
        }
    });

    // Render dataFrame
    dataFrames = document.querySelectorAll(".dataframe")

    dataFrames.forEach(df =>{
        df.parentElement.classList = 'df';
    })

    // Render math formulas
    renderMathInElement(document.body, {
        delimiters: [
            {left: "\\[", right: "\\]", display: true},
            {left: "$$", right: "$$", display: true},
            {left: "\\(", right: "\\)", display: false},
            {left: "$", right: "$", display: false}
        ]
    });

    // Add blank target in anchor rag
    document.querySelectorAll('a').forEach(a =>{
        a.setAttribute('target', 'blank')
    })    
    // Highlight code
    hljs.highlightAll();
}

main();