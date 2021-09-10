const form = get(".inputarea")
const input = get(".forminput")
const chat = get(".chat")

const BOT_NAME = "Altruist Bot"

async function init() {
	//The first phase of the chat flow
	const firstPhase  = flow['name']

	storeLocally('phaseID', 'name')
	storeLocally('name', 'Username')

	const text = firstPhase.question()
	answer(text)
}

function appendMessage(name, side, text, phaseID) {
	//If the text is not an image then use <p>
	if (!text.includes('img')) {
		text = `<p>${text.split('\n').join('</p><p>')}</p>`
	}

	//   Simple solution for small apps
	const msgHTML = `
		<div class="msg ${side}-msg">
		<div class="msg-img"></div>

		<div class="msg-bubble">
			<div class="msg-info">
			<div class="msg-info-name">${name}</div>
			<div class="msg-info-time">${formatDate(new Date())}</div>
			</div>

			<div class="msg-text" id="${phaseID}">${text}</div>
		</div>
		</div>
	`

	chat.insertAdjacentHTML("beforeend", msgHTML)
	chat.scrollTop += 500
}

//Function that append a Bot's message
function answer(text) {
	// const delay = text.split(" ").length * 100
	//setTimeout(() => {
	appendMessage(BOT_NAME, "left", text, "bot")
	//}, delay)
}


// Utils
function get(selector, root = document) {
  return root.querySelector(selector)
}

function formatDate(date) {
	const h = "0" + date.getHours()
	const m = "0" + date.getMinutes()

	return `${h.slice(-2)}:${m.slice(-2)}`
}

function random(min, max) {
  return Math.floor(Math.random() * (max - min) + min)
}

form.addEventListener("submit", async (event) => {
	event.preventDefault()

	const msgText = input.value
	if (!msgText) return

	const name = fetchFromStorage('name')
	const phaseID = fetchFromStorage('phaseID')
	const oldPhase = flow[phaseID]

	console.log(phaseID, oldPhase)

	//Append user's message
	appendMessage(name, "right", msgText, phaseID)
	input.value = ''

	//Answer to user
	result = await oldPhase.handler(msgText)
	console.log(oldPhase);
	//For testing
	console.log(`For phase: ${phaseID}`, `handling`, msgText)
	console.log(`Result:`, result)
	answer(result.answer)

	if (result.next) {
		storeLocally('phaseID', result.next)
		const newPhase = flow[result.next]

		const question = await newPhase.question()
		answer(question)
	}
	else {
		answer("Bye Bye!")
	}

})

init()
