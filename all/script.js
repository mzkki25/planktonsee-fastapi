$(document).ready(function() {
    const $btn = $('.btn');
    const $start = $('.btn-start');
    const $home = $('.btn-back');

    $btn.hover(
        function() {
            var svg = $(this).find('svg path');
            $(this).data('timeout', setTimeout(function() {
                svg.css('fill', '#ffffff'); 
            }, 100)); 
        }, 
        function() {
            clearTimeout($(this).data('timeout')); 
            $(this).find('svg path').css('fill', '#59a9d4');
        }
    );

    $start.on('click', function() {
        window.location.href = '/predict_ui'; 
    });

    $home.on('click', function() {
        window.location.href = '/'; 
    });
    
});

$(document).ready(function () {
    let selectedFile = null;

    $("#file-image-upload").change(function (e) {
        const file = e.target.files[0];
        if (file) {
            selectedFile = file;
            $("#image-upload").val(file.name);
        }
    });

    $(".btn-predict-image").click(function () {
        if (!selectedFile) {
            swal({
                text: "Silakan upload gambar terlebih dahulu.",
                icon: "error",
                customClass: {
                    popup: 'custom-swal-popup',
                    title: 'custom-swal-title',
                    content: 'custom-swal-content',
                    confirmButton: 'custom-swal-button'
                }
            });            
            return;
        }

        const modelOption = $("#deep-learning-model").val();
        const llmOption = $("#llm-model").val();

        const formData = new FormData();

        formData.append("img_path", selectedFile);
        formData.append("model_option", modelOption);
        formData.append("llm_option", llmOption);

        $("#transparant-bg").show();
        $("#load").show();

        fetch("/predict_action", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            $("#load").hide();
            $("#transparant-bg").hide();

            if (data.error) {
                swal({
                    title: "Error",
                    text: "Terjadi kesalahan pada data.",
                    icon: "error",
                    customClass: {
                        popup: 'custom-swal-popup',
                        title: 'custom-swal-title',
                        content: 'custom-swal-content',
                        confirmButton: 'custom-swal-button'
                    },
                });
                return;
            }

            window.location.href = `/result/${data.doc_id}`;
        })
        .catch(err => {
            $("#load").hide();
            $("#transparant-bg").hide();

            swal({
                title: "Error",
                text: "Terjadi kesalahan saat memprediksi gambar.",
                icon: "error",
                customClass: {
                    popup: 'custom-swal-popup',
                    title: 'custom-swal-title',
                    content: 'custom-swal-content',
                    confirmButton: 'custom-swal-button'
                },
            });
            console.error("Error:", err);
        });
    });
});
