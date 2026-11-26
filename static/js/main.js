const navToggle = document.querySelector(".nav-toggle");
const navigation = document.querySelector("#navigation");
const compact = window.matchMedia("(max-width: 700px)");

function collapseNavigation() {
    navToggle.hidden = !compact.matches;
    navToggle.setAttribute("aria-expanded", "false");
    navigation.toggleAttribute("data-collapsed", compact.matches);
}

collapseNavigation();
compact.addEventListener("change", collapseNavigation);
navToggle.addEventListener("click", () => {
    const collapsed = navigation.toggleAttribute("data-collapsed");
    navToggle.setAttribute("aria-expanded", String(!collapsed));
});

const labForm = document.querySelector("[data-lab-form]");
if (labForm) {
    const conditionalControls = [...labForm.querySelectorAll("[data-show-for]")];
    const result = document.getElementById("results");
    const running = labForm.querySelector("[data-running]");
    const retained = labForm.querySelector("[data-retained]");

    function updateControls() {
        for (const section of conditionalControls) {
            const [field, values] = section.dataset.showFor.split(":");
            const visible = values.split(",").includes(labForm.elements[field].value);
            section.hidden = !visible;
            for (const control of section.querySelectorAll("input, select")) {
                control.disabled = !visible;
            }
        }
    }

    updateControls();
    labForm.addEventListener("change", updateControls);
    for (const upload of labForm.querySelectorAll("[data-upload]")) {
        upload.addEventListener("change", () => {
            if (upload.files.length) {
                labForm.elements["sample_" + upload.dataset.upload].checked = false;
            }
        });
    }

    labForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const data = new FormData(labForm, event.submitter);
        const buttons = [...labForm.querySelectorAll("button[type=submit]")];
        buttons.forEach((button) => { button.disabled = true; });
        running.hidden = false;
        result.setAttribute("aria-busy", "true");
        try {
            const response = await fetch(labForm.action, {method: "POST", body: data});
            const page = new DOMParser().parseFromString(await response.text(), "text/html");
            const incoming = page.getElementById("results");
            if (!incoming) throw new Error("Response did not contain a result panel.");
            result.replaceChildren(...incoming.childNodes);
            if (response.ok) {
                const token = page.querySelector("[name=previous_id]").value;
                labForm.elements.previous_id.value = token;
                for (const upload of labForm.querySelectorAll("[data-upload]")) {
                    if (upload.files.length && !upload.disabled) {
                        upload.value = "";
                        labForm.elements["sample_" + upload.dataset.upload].checked = false;
                        retained.hidden = false;
                    }
                }
            }
        } catch {
            let error = result.querySelector(".network-error");
            if (!error) {
                error = document.createElement("p");
                error.className = "error network-error";
                error.setAttribute("role", "alert");
                result.prepend(error);
            }
            error.textContent = "The server did not return a result. Check your connection and try again.";
        } finally {
            running.hidden = true;
            buttons.forEach((button) => { button.disabled = false; });
            result.removeAttribute("aria-busy");
            updateControls();
            result.focus();
        }
    });
}
