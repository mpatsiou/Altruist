const form = get(".inputarea")
const input = get(".forminput")
const chat = get(".chat")

// Icons made by Freepik from www.flaticon.com
const BOT_IMG = "https://image.flaticon.com/icons/svg/327/327779.svg"
const PERSON_IMG = "https://image.flaticon.com/icons/svg/145/145867.svg"

const BOT_NAME = "Altruist Bot"
const PERSON_NAME = "Anonymous"

hello()

function botResponse(text) {
	let response

	return answer('test')
}

form.addEventListener("submit", event => {
	event.preventDefault()

	const msgText = input.value
	if (!msgText) return

	appendMessage(PERSON_NAME, PERSON_IMG, "right", msgText)
	input.value = ""

	text = botResponse(msgText)
	answer(text)
})

function appendMessage(name, img, side, text) {
	//   Simple solution for small apps
	const msgHTML = `
		<div class="msg ${side}-msg">
		<div class="msg-img" style="background-image: url(${img})"></div>

		<div class="msg-bubble">
			<div class="msg-info">
			<div class="msg-info-name">${name}</div>
			<div class="msg-info-time">${formatDate(new Date())}</div>
			</div>

			<div class="msg-text">${text}</div>
		</div>
		</div>
	`

	chat.insertAdjacentHTML("beforeend", msgHTML)
	chat.scrollTop += 500
}

function answer(text) {
	const delay = text.split(" ").length * 100
	setTimeout(() => {
			appendMessage(BOT_NAME, BOT_IMG, "left", text)
	}, delay)
}

function hello() {
	const text = `My name is ${BOT_NAME}.
		Let me know if you have any questions regarding our tool!
		What's your name?`

	answer(text)
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