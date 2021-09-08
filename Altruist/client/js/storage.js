const storage = window.localStorage

function storeLocally(key, data) {
    storage.setItem(key, JSON.stringify(data))
}

function fetchFromStorage(key) {
    return JSON.parse(storage.getItem(key))
}
