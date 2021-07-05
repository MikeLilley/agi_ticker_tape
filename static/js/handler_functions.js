function query_button(){
  document.getElementById("Result").value = "Computing...";

  $.ajax({
  type: "POST",
  url: "/query_response",
  data: document.getElementById("Question").value,
  success: (function(out) {
    document.getElementById("Result").value = JSON.parse(out).response; })
})};

function query_file_button(){
  console.log("button was clicked");
};

function new_knowledge()
{
  if(typeof(document.getElementById('file_input').files[0]) === "undefined" && document.getElementById('website_input').value !== "")
  {
    $.ajax({
    type: "POST",
    url: "/scrape_website",
    data: document.getElementById("website_input").value,
    success: (function(out) {
      document.getElementById("knowledge_database").innerHTML = JSON.parse(out).table; })})
  }

  else if(typeof(document.getElementById('file_input').files[0]) !== "undefined" && document.getElementById('website_input').value === "")
  {
    var file_data = new FormData($('#upload_knowledge')[0]);
    document.getElementById("new_knowledge_status").value = "Processing File...";

    $.ajax({
    type: "POST",
    processData: false,
    contentType: false,
    url: "/upload_file",
    data: file_data,
    success: (function(out) {
      get_database_table();
      document.getElementById("new_knowledge_status").value = JSON.parse(out).response; })
  })
  }

  else if(typeof(document.getElementById('file_input').files[0]) !== "undefined" && document.getElementById('website_input').value !== "")
  {
    document.getElementById("new_knowledge_status").value = "Please Only Input One Type of Data";
  }

  else
  {
    document.getElementById("new_knowledge_status").value = "Upload Invalid";
  }

};

function get_database_table(){
  $.ajax({
  type: "POST",
  url: "/return_table",
  data: document.getElementById("table_select").value,
  success: (function(out) {
    document.getElementById("knowledge_database").innerHTML = JSON.parse(out).table; })
})};

//INITIALIZATION FUNCTIONS TO RUN ON STARTUP
$(document).ready( function () {
    $('#knowledge_database').DataTable();
    console.log(document.getElementById('file_input').files[0])
} );
