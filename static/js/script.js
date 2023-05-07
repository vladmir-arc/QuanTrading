var myButton = document.getElementById("myButton");

myButton.addEventListener("click", function() {
    alert("You tapped me!");
});

function getSelectedItems() {
    const select = document.getElementById("mySelect");
    const selectedItems = [];

    for (let i = 0; i < select.options.length; i++) {
        if (select.options[i].selected) {
            selectedItems.push(select.options[i].value);
        }
    }

    alert(selectedItems);
}