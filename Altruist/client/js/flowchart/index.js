const URL = 'http://localhost:3000'

const flow = {
    'name': {
        question: askName,
        handler: handleName
    },
    'prediction': {
        question: askPrediction,
        handler: handlePrediction
    },
    'details': {
        question: askDetails,
        handler: handleDetails
    },
    'go_back': {
        question: askGoBack,
        handler: handleGoBack
    },
    'altruist':  {
        question: askAltruist,
        handler: handlerAlruist
    },
    'go_back2': {
        question: askGoBack2,
        handler: handleGoBack2
    },
    'done': {
        question: askDone,
        handler: handleDone
    }
}

function askDone() {
    return "Wanna enter new features?"
}

function handleDone(answer) {
    if (answer.startsWith('y')) {
        return {
            answer: "Let's go!",
            next: 'prediction'
        }
    }

    return {
        answer: "See yuh around!",
        next: null
    }
}

function askName() {
    return "Hello, what's your name?"
}

async function handleName(name) {
    storeLocally('name', name)

    return {
        answer: "Nice to meet you! Let's make some prediction.",
        next: 'prediction'
    }
}

async function askPrediction() {
    const featureNames = await request('get', '/features_names')
    storeLocally('featureNames', featureNames)

    const parsedFeatures = featureNames.map(s => "* " + s).join('\n')
    console.log("parsedFeatures\n", parsedFeatures);
    console.log("featureNames", featureNames)

    return "Fill in the following features:\n" + parsedFeatures
}

async function handlePrediction(featureValues) {
    const values = featureValues.split(' ').map(Number)
    const features = {}

    if (values.length != fetchFromStorage('featureNames').length) {
        return {
            answer: "Please give correct feature values",
            next: "prediction"
        }

    }

    i = 0
    for (const name of fetchFromStorage('featureNames')) {
        console.log('Feature name', name, values[i]);
        features[name] = values[i]
        ++i
    }

    storeLocally('features', features)
    const prediction = await request('post', '/predict', features)

    return {
        answer: `I got the features! It seems that this is a: <b>${prediction}</b>`,
        next: 'details'
    }
}

function askGoBack() {
    return "Do you want to go back?"
}

function handleGoBack(answer) {
    answer = answer.toLowerCase()
    if (answer == 'yes' || answer == 'y') {
        return {
            answer: "Going back!",
            next: 'details'
        }
    }
    else if (answer == 'no' || answer == 'n') {
        return {
            answer: "Alright!",
            next: 'altruist'
        }
    }

    return { answer: "Incomprehensible answer!", next: 'go_back' }
}

async function askDetails() {
    return `
	Now you can see the interpretation of some models, as well as the information about them.
	1)information about LIME
	2)information about Shap
	3)information about Permutation Importance(PI)
	4)interpretation of LIME
	5)interpretation of Shap
	6)interpretation of Permutation Importance(PI)
	`
}

async function handleDetails(answer) {
    const options = ['info_lime', 'info_shap', 'info_pi', 'interpret_lime', 'interpret_shap', 'interpret_pi']
    // var {synonyms} =  require("synonyms")
    // synonyms('information', "n")
    // console.log(synonyms.prediction)

    const fuse = new Fuse(options, {includeScore: true})
    const result = fuse.search(answer)[0]

    if (!result || result.score > 0.6) {
        return {
            answer: "Invalid options! Answer only some of the specific options",
            next: 'details'
        }
    }

    let text = ''
    switch (result.item) {
        //information cases
        case 'info_lime':
            text = "info about lime.."
            break
        case 'info_shap':
            text = "info about shap.."
            break
        case 'info_pi':
            text = "info about pi.."
            break
        default:
        //Interpret cases
        const [_, fi_method] = result.item.split('_')
        const features = fetchFromStorage('features')
        const values = Object.values(features).join(',')

        const fi = await request('get', `/feature_importance?method=${fi_method}&values=${values}`)
        const featureNames = fetchFromStorage('featureNames')
        const barplotData = zipObject(featureNames, fi)

        const plot = generateBarplot(fi_method, barplotData)

        var img = document.createElement('img')
        img.src = plot
        img.style="width: 100%; height: 500px;"

        text = img.outerHTML
    }

    return {
        answer: `You chose: ${result.item}\n${text}`,
        next: 'go_back'
    }
}

async function askAltruist() {
    return `
    After that, I suggest you to use the Altruist.
    A new methodology that aims to tackle a few problems of feature importance-based aproaches.
    You can see:
    1)Info of Altruist
    2)Combinatorial interpretation from Altruist
    3)The untruthful features of LIME
    4)The untruthful features of Shap
    5)The untruthful features of Permutation Importance(PI)
    6)The counterfactuals for every feature
    7)Go to previous step
    `
}

async function handlerAlruist(answer) {
    const options = ['info_altruist', 'interpret_altruist', 'untruthful_features_lime', 'untruthful_features_shap', 'untruthful_features_pi', 'counterfactuals', 'previous_phase']

    //na diavasw ksana fuse kai pws to xrisimopoiw
    const fuse = new Fuse(options, {includeScore: true})
    const result = fuse.search(answer)[0]
    if (!result || result.score > 0.6) {
        return {
            answer: "Invalid options! Answer only some of the specific options",
            next: 'altruist'
        }
    }


    let next = 'go_back2'
    const features = fetchFromStorage('features')
    const values = Object.values(features).join(',')
    const explanation = await request('get', `/altruist?values=${values}`)
    const [untruthful, counterfactuals] = explanation

    let fi_method = ''

    let text= ''
    switch (result.item) {
        case 'info_altruist':
            text = "info about altruist"
            break;
        case 'interpret_altruist':
            text = "interpretation of altruist"
            //To do the plot method
            break;
        case 'untruthful_features_lime':
            fi_method = result.item.split('_').pop()
            text = "Untruthful features of " + fi_method + " : " + untruthful[0] + " (" + untruthful[0].length + ")"
            break;
        case 'untruthful_features_shap':
            fi_method = result.item.split('_').pop()
            text = "Untruthful features of " + fi_method + " : " + untruthful[1] + " (" + untruthful[1].length + ")"
            break;
        case 'untruthful_features_pi':
            fi_method = result.item.split('_').pop()
            text = "Untruthful features of " + fi_method + " : " + untruthful[2] + " (" + untruthful[2].length + ")"
            break;
        case 'counterfactuals':
            const lengths = untruthful.map(l => l.length)
            const featureNames = fetchFromStorage("featureNames")

            const minIdx = lengths.indexOf(Math.min(...lengths))
            const counterfactualFeature = featureNames[counterfactuals[minIdx][0][0] - 1]

            text = minIdx > -1 ?
                'The counterfactuals for the feature: ' + counterfactualFeature + " is " + counterfactuals[minIdx][0][1] :
                'No counterfactuals'
            break;
        case 'previous_phase':
            text = "previous phase"
            next = "details"
           break;
        default:
            text = 'testing'
    }
    return {
        answer: `You chose: ${result.item}\n${text}`,
        next: next
    }
}

function askGoBack2() {
    return "Do you want to go back?"
}

function handleGoBack2(answer) {
    answer = answer.toLowerCase()
    if (answer == 'yes' || answer == 'y') {
        return {
            answer: "Going back!",
            next: 'altruist'
        }
    }
    else if (answer == 'no' || answer == 'n') {
        return {
            answer: "Alright!",
            next: 'done'
        }
    }

    return { answer: "Incomprehensible answer!", next: 'go_back' }
}

// Function that talks to server
async function request(method, endpoint, data = null) {
    const response = await axios[method](URL + endpoint, data)
    return response.data
}


// data: {'variance': 0.2, 'cyrtosis': -0.12, ..}
const generateBarplot = (title, data) => {
    const dataPoints = []
    for (const key in data) {
        dataPoints.push({ label: key, y: data[key] })
    }

    const div = document.createElement('div')
    div.id = "chart_hidden"
    div.style="width: 0%; height: 0; position: fixed;"
    document.body.appendChild(div);

    const chart = new CanvasJS.Chart("chart_hidden", {
        animationEnabled: false,
        theme: "light2",
        title: {
            text: title
        },
        data: [{
            type: "column",
            showInLegend: true,
            legendMarkerColor: "grey",
            dataPoints: dataPoints
        }]
    });

    chart.render()
    const base64Image = chart.canvas.toDataURL();

    const elem = document.getElementById("chart_hidden");
    elem.remove();

    return base64Image
}

function zipObject(keys, values) {
    const obj = {};

    keys.forEach((key, index) => {
      obj[key] = values[index];
    })

    return obj;
}
