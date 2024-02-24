function collapse_menu(){
    is_shown = $("#sidebar").hasClass("show");
    console.log(is_shown)
    if (is_shown==true){
        $("#collapse_menu_button").css("margin-left", "0px")
        $("#sidebar").removeClass("show")
        $("#collapse_icon_1").removeClass("fa-chevron-left")
        $("#collapse_icon_1").addClass("fa-chevron-right")

        $("#collapse_icon_2").removeClass("fa-chevron-left")
        $("#collapse_icon_2").addClass("fa-chevron-right")

    }
    else {
        $("#collapse_menu_button").css("margin-left", "-18px")
        $("#sidebar").addClass("show")
        $("#collapse_icon_1").removeClass("fa-chevron-right")
        $("#collapse_icon_1").addClass("fa-chevron-left")

        $("#collapse_icon_2").removeClass("fa-chevron-right")
        $("#collapse_icon_2").addClass("fa-chevron-left")
    }
}