/* utils.js
 * 
 * This file contains helper utility functions/objects for use by the
 * main experiment code.
 */


// Object to hold the state of the experiment. It is initialized to
// reflect the hash in the URL (see set_hash and load_hash for
// details).
var State = function () {

    // 0 (false) or 1 (true)
    this.instructions;
    // Trial index
    this.index;
    // One of the phases defined in TRIAL
    this.trial_phase;

    // Update the instructions flag. Defaults to 1.
    this.set_instructions = function (instructions) {
        if (instructions != undefined) {
            this.instructions = instructions;
        } else {
            this.instructions = 1;
        }
    };

    // Update the trial index. Defaults to 0.
    this.set_index = function (index) {
        if (index != undefined) {
            this.index = index;
        } else {
            this.index = 0;
        }
    };

    // Update the trial phase. Defaults to TRIAL.prestim.
    this.set_trial_phase = function (trial_phase) {
        if (!this.instructions) {
            if (trial_phase != undefined) {
                this.trial_phase = trial_phase;
            } else {
                this.trial_phase = TRIAL.prestim;
            }
        } else {
            this.trial_phase = undefined;
        }
    };

    // Set the URL hash based on the current state. If
    // this.instructions is 1, then it will look like:
    //
    //     <instructions>-<index>
    // 
    // Otherwise, if this.instructions is 0, it will be:
    //
    //     <instructions>-<index>-<trial_phase>
    //
    // Returns the URL hash string.
    this.set_hash = function () {
        var parts = [
            this.instructions,
            this.index
        ];

        if (!this.instructions) {
            parts[parts.length] = this.trial_phase;
        }

        var hash = parts.join("-");
        window.location.hash = hash;
        return hash;
    };

    // Update the State object based on the URL hash
    this.load_hash = function () {
        // get the URL hash, and remove the # from the front
        var hash = window.location.hash.slice(1);

        if (window.location.hash == "") {
            // no hash is present, so use the defaults
            this.set_instructions();
            this.set_index();
            this.set_trial_phase();

        } else {
            // split the hash into its components and set them
            var parts = hash.split("-").map(
                function (item) {
                    return parseInt(item);
                });
            this.set_instructions(parts[0]);
            this.set_index(parts[1]);
            this.set_trial_phase(parts[2]);
        }
    };

    // Return a list of the state's properties in human-readable form,
    // to be recorded as data in the database
    this.as_data = function () {
        var instructions = Boolean(this.instructions);
        var index = this.index;
        var trial_phase;
        
        // Find the name of the trial phase (or just use
        // an empty string if instructions is true)
        if (!instructions) {
            for (item in TRIAL) {
                if (TRIAL[item] == this.trial_phase) {
                    trial_phase = item;
                    break
                }
            }
        } else {
            trial_phase = "";
        }

        return {
            'instructions': instructions,
            'index': index, 
            'trial_phase': trial_phase
        };
    };

    // Initialize the State object components
    this.load_hash();
};

// Object to properly format rows of data
var DataRecord = function () {
    this.fields = [
        "instructions",
        "index",
        "trial_phase",
        "feedback",
        "ratio",
        "counterbalance",
        "stimulus",
        "flipped",
        "theta",
        "response",
        "response_time"
    ];

    this.update = function (other) {
        _.extend(this, other);
    };

    this.to_array = function () {
        var arr = [];
        for (i in this.fields) {
            arr.push(this[this.fields[i]]);
        }
        return arr;
    };
};

// Log a message to the console, if debug mode is turned on.
function debug(msg) {
    if ($c.debug) {
        console.log(msg);
    }
}

// Throw an assertion error if a statement is not true.
function AssertException(message) { this.message = message; }
AssertException.prototype.toString = function () {
    return 'AssertException: ' + this.message;
};
function assert(exp, message) {
    if (!exp) {
        throw new AssertException(message);
    }
}

// Open a new window and display the consent form
function open_window(hitid, assignmentid, workerid) {
    popup = window.open(
        '/consent?' + 
            'hitId=' + hitid + 
            '&assignmentId=' + assignmentid + 
            '&workerId=' + workerid,
        'Popup',
        'toolbar=no,' +
            'location=no,' +
            'status=no,' +
            'menubar=no,' + 
            'scrollbars=yes,' + 
            'resizable=no,' + 
            'width=' + screen.availWidth + ',' +
            'height=' + screen.availHeight + '');
    popup.onunload = function() { 
        location.reload(true) 
    };
}

// Update the progress bar based on the current trial and total number
// of trials.
function update_progress(num, num_trials) {
    debug("update progress");
    var width = 2 + 98 * (num / (num_trials - 1.0));
    $("#indicator-stage").css({"width": width + "%"});
    $("#progress-text").html("Progress " + (num + 1) + "/" + num_trials);
}

// Show new_elems, and then hide everything else
function show_phase(new_elem, callback) {
    $(".phase").removeClass("current-phase");
    $(".phase").addClass("inactive-phase");

    $("#" + new_elem).removeClass("inactive-phase");
    $("#" + new_elem).addClass("current-phase");

    $(".current-phase").show();
    $(".inactive-phase").hide();
    if (callback) callback();
}

function scale_vertex(xy, canvas) {
    var x = (xy[0] * (canvas.width / 2.)) + (canvas.width / 2.);
    var y = (xy[1] * -(canvas.height / 2.)) + (canvas.height / 2.);
    return [x, y];
}

function draw_shape(canvas, vertices) {
    var ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 3;
    ctx.beginPath();

    var v;
    for (i in vertices) {
        v = scale_vertex(vertices[i], canvas);
        if (i == 0) {
            ctx.moveTo(v[0], v[1]);
        } else {
            ctx.lineTo(v[0], v[1]);
        }
    }
    
    ctx.closePath();
    ctx.stroke();
}

function draw_fixation(canvas) {
    var ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 5;
    ctx.beginPath();

    var w2 = canvas.width / 2.0;
    var h2 = canvas.height / 2.0;
    var x = 10;

    ctx.moveTo(w2 - x, h2);
    ctx.lineTo(w2 + x, h2);

    ctx.moveTo(w2, h2 - x);
    ctx.lineTo(w2, h2 + x);
    
    ctx.stroke();
}
