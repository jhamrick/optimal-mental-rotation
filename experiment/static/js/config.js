/* config.js
 * 
 * This file contains the code necessary to load the configuration
 * for the experiment.
 */

// Enum-like object mapping trial phase names to ids, in the order
// that the phases should be presented.
var TRIAL = Object.freeze({
    prestim: 0,
    stim: 1,
    feedback: 2,
    length: 3
});

// Enum-like object for representing key names.
var KEYS = new Object();
KEYS[TRIAL.prestim] = {
    66: ""  // b
};
KEYS[TRIAL.stim] = {
    83: "same",      // s
    68: "different", // d
};
KEYS[TRIAL.feedback] = {};

// Object to hold the experiment configuration. It takes as parameters
// the numeric codes representing the experimental condition and
// whether the trials are counterbalanced.
var Config = function (condition, counterbalance) {

    // These are the condition and counterbalancing ids
    this.condition = condition;
    this.counterbalance = counterbalance;

    // Whether debug information should be printed out
    // TODO: set debug to false
    this.debug = true;
    // The amount of time to fade HTML elements in/out
    this.fade = 200;
    // List of trial information object for each experiment phase
    this.trials = null;

    // Canvas width and height
    this.canvas_width = 300;
    this.canvas_height = 300;

    // Lists of pages and examples for each instruction page.  We know
    // the list of pages we want to display a priori.
    this.instructions = {
        pages: ["instructions"]
    };

    // The list of all the HTML pages that need to be loaded
    this.pages = [
        "trial.html", 
        "submit.html"
    ];

    // Parse the JSON object that we've requested and load it into the
    // configuration
    this.parse_config = function (data) {
        this.trials = _.shuffle(data["trials"]);

        console.log(data.examples);
        this.instructions.examples = [data.examples];
    };

    // Load the experiment configuration from the server
    this.load_config = function () {
        var that = this;
        $.ajax({
            dataType: "json",
            url: "/static/json/" + this.condition + 
                "-cb" + this.counterbalance + ".json",
            async: false,
            success: function (data) { 
                if (that.debug) {
                    console.log("Got configuration data");
                }
                that.parse_config(data);
            }
        });
    };

    // Request from the server configuration information for this run
    // of the experiment
    this.load_config();
};
