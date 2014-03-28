# Simulation logic

1. The fileserver keeps a copy of tasks.json and completed.json, and
   sends over tasks (one at a time) to the clients.
   
2. First, the client requests the simulation root directory from the
   server with `panda_connect`.

3. Then, it loops over tasks from the server until there are none
   left. It requests a task from the server with `panda_request`,
   which will be either a JSON object of task parameters, or `None` if
   there are no more tasks.

4. When the client is finished with a task, it will zip up the data,
   `scp` it over to the server, and then tell the server to extract
   the data with `panda_extract`. Finally, it will tell the server to
   mark it as done with `panda_complete`.
