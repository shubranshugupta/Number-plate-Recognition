const Toast = Swal.mixin({
    toast: true,
    position: 'top-end',
    showConfirmButton: false,
    timer: 4000,
    timerProgressBar: true,
    didOpen: (toast) => {
        toast.addEventListener('mouseenter', Swal.stopTimer)
        toast.addEventListener('mouseleave', Swal.resumeTimer)
    }
})

function checkSize(file) {
    const fsize =  Math.round(((file.size) / 1024));
    if (fsize >= 3072) {
        return false;
    }
    return true;
}

function blurBackgroundOnLoading() {
    let heading = document.getElementById("heading");
    heading.style["filter"] = "blur(3px)";

    let uplodField = document.getElementById("uplodField");
    uplodField.style["filter"] = "blur(3px)";
    uplodField.disabled = "true";

    let submitBtn = document.getElementById("submitBtn");
    submitBtn.style["filter"] = "blur(3px)";
    submitBtn.disabled = "true";

    // let cameraBtn = document.getElementById("cameraBtn");
    // cameraBtn.style["filter"] = "blur(3px)";
    // cameraBtn.disabled = "true";

    document.getElementById("loader-wrapper").style.display = "block";
}

function submitFile() {
    let data = new FormData();
    let file = document.getElementById("imageUpload").files[0];
    if (file.name.endsWith('.png') || file.name.endsWith('.jpg') || file.name.endsWith('.jpeg')) {

        setTimeout(blurBackgroundOnLoading, 4000);

        Toast.fire({
            icon: 'success',
            title: 'Photo Uploaded successfully!!'
        })

        data.append('image', file, file.name);

        let xhr = new XMLHttpRequest();
        let url = "/uploadImage";
        xhr.open('POST', url, true);
        xhr.setRequestHeader("Access-Control-Allow-Origin", "*");
        xhr.send(data);
        xhr.onreadystatechange = function () {
            if (xhr.readyState === 4 && xhr.status === 200) {
                let statusReturn = JSON.parse(this.responseText)['status'];
                if (statusReturn === true) {
                    location.replace(JSON.parse(this.responseText)['redirect']);
                }
                else {
                    document.getElementById("loader-wrapper").style.display = "block";
                    Toast.fire({
                        icon: 'error',
                        title: 'Internal Error'
                    })
                }
            }
        }
    }
    else if (file.name.endsWith('.mp4') || file.name.endsWith('.avi')) {
        if (checkSize(file)){
            setTimeout(blurBackgroundOnLoading, 4000);

            Toast.fire({
                icon: 'success',
                title: 'Video Uploaded successfully!!'
            })

            data.append('video', file, file.name);

            let xhr = new XMLHttpRequest();
            let url = "/uploadVideo";
            xhr.open('POST', url, true);
            xhr.setRequestHeader("Access-Control-Allow-Origin", "*");
            xhr.send(data);

            xhr.onreadystatechange = function () {
                if (xhr.readyState === 4 && xhr.status === 200) {
                    location.replace(JSON.parse(this.responseText)['redirect']);
                }
            }
        }
        else {
            Toast.fire({
                icon: 'warning',
                title: 'Video File size should be < 3mb'
            })
        }
    }
    else {
        Toast.fire({
            icon: 'warning',
            title: 'Plz!! Upload proper file'
        })
    }
}

function cameraClick(){
    let xhr = new XMLHttpRequest();
    let url = "/uploadVideo";
    xhr.open('POST', url, true);
    xhr.setRequestHeader("Access-Control-Allow-Origin", "*");
    xhr.send();

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            location.replace(JSON.parse(this.responseText)['redirect']);
        }
    }
}

function stopBacktoHomepage(){
    let xhr = new XMLHttpRequest();
    let url = "/toMain";
    xhr.open('GET', url, true);
    xhr.setRequestHeader("Access-Control-Allow-Origin", "*");
    xhr.send();

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            location.replace(JSON.parse(this.responseText)['redirect']);
        }
    }
}