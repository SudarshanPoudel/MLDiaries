modules = [
    {
        "name": "Regression and Classification",
        "filename": "01_Regression.md"
    },
    {
        "name": "Popular ML Algorithms",
        "filename": "02_Popular_ML_Algorithms.md"
    },
    {
        "name": "Ensemble Methods",
        "filename": "03_Ensemble_Methods.md",
    },
    {
        "name": "Clustering",
        "filename": "04_Clustering.md",
    },
    {
        "name": "Neural Networks",
        "filename": "05_Neural_Networks.md"
    },
    {
        "name": "Image Processing",
        "filename": "06_Image_Processing.md"
    },
    {
        "name": "Convolutional Neural Networks",
        "filename": "07_Convolutional_Neural_Networks.md"
    },
    {
        "name": "Transfer Learning and Autoencoders",
        "filename": "08_Transfer_Learning_and_Autoencoders.md"
    },
    {
        "name": "Natural Language Processing",
        "filename": "09_Natural_Language_Processing.md"
    },
    {
        "name": "Recurrent Neural Networks",
        "filename": "10_Recurrent_Neural_Networks.md"
    },
    {
        "name": "Transformers",
        "filename": "11_Transformers.md"
    },
    {
        "name": "Language Models and LLMs",
        "filename": "12_Language_Models_and_LLMs.md"
    },
    {
        "name": "Foundational Models",
        "filename": "13_Foundational_Models.md"
    }
]


projects = [
    {
        "name": "Flower classification with NN",
        "filename": "flower_classification.md"
    },
]
papers = [
    {
        "name": "Attention is all you need",
        "filename": "attention.md"
    },
]


// Display chapters
// Get the URL search parameters
const searchParams = new URLSearchParams(window.location.search);

// Get the id parameter value
let type = searchParams.get('type');


if (type == "project") {
    arr = projects;
    folder = "Projects";
} else if (type == "paper") {
    arr = papers;
    folder = "Papers";
} else {
    type = 'module';
    arr = modules;
    folder = "Modules";
}


document.querySelector('.title').innerHTML = folder

index = 1;
const formatNumber = num => num >= 0 && num < 10 ? `0${num}` : `${num}`;

chapterGrid = document.querySelector('.chapter-grid');
chapterGrid.innerHTMl = ''
arr.forEach((a) => {
    btn = document.createElement('a')
    btn.href = `../Content/content.html?type=${type}&id=${index}`
    btn.className = 'module'
    btn.innerHTML = `<div class="num">${formatNumber(index)}</div><div class="name">${a['name']}</div>`
    chapterGrid.appendChild(btn)
    index++;
});


