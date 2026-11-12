using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;

internal sealed class RatelLauncher : Form
{
    private readonly string repository;
    private readonly TextBox video = new TextBox();
    private readonly TextBox config = new TextBox();
    private readonly TextBox output = new TextBox();
    private readonly ComboBox sessionMode = new ComboBox();
    private readonly ComboBox speed = new ComboBox();
    private readonly Button launch = new Button();
    private readonly Label hint = new Label();
    private readonly Button outputBrowse;
    private readonly Label status = new Label();

    [STAThread]
    private static void Main(string[] args)
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        try
        {
            Application.Run(new RatelLauncher(args));
        }
        catch (Exception error)
        {
            MessageBox.Show(error.Message, "RATEL launcher", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }

    private static string FindRepository()
    {
        string[] candidates = { AppDomain.CurrentDomain.BaseDirectory,
                                "D:\\Code\\Python\\BehaviourExperiments" };
        foreach (string candidate in candidates)
        {
            if (File.Exists(Path.Combine(candidate, ".venv-oft", "Scripts", "pythonw.exe")) &&
                File.Exists(Path.Combine(candidate, "src", "behaviour", "manual_tracking.py")))
                return Path.GetFullPath(candidate);
        }
        throw new DirectoryNotFoundException("Cannot find RATEL and its .venv-oft environment. " +
            "Place this launcher in the BehaviourExperiments folder with its installed environment.");
    }

    private RatelLauncher(string[] args)
    {
        repository = FindRepository();
        Text = "RATEL - Manual tracking";
        Font = new Font("Segoe UI", 10);
        AutoScaleMode = AutoScaleMode.Dpi;
        ClientSize = new Size(760, 390);
        MinimumSize = new Size(700, 425);
        StartPosition = FormStartPosition.CenterScreen;
        Icon = SystemIcons.Application;

        TableLayoutPanel layout = new TableLayoutPanel();
        layout.Dock = DockStyle.Fill;
        layout.Padding = new Padding(18);
        layout.ColumnCount = 3;
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 115));
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 100));
        layout.RowCount = 9;
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 48));
        for (int i = 1; i <= 5; i++)
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 37));
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 35));
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        Controls.Add(layout);

        Label intro = new Label();
        intro.Text = "Follow the rat with your pen or mouse. Choose a video to start or resume.";
        intro.Dock = DockStyle.Fill;
        layout.Controls.Add(intro, 0, 0);
        layout.SetColumnSpan(intro, 3);

        layout.Controls.Add(new Label { Text = "Session", Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft }, 0, 3);
        sessionMode.DropDownStyle = ComboBoxStyle.DropDownList;
        sessionMode.Items.AddRange(new object[] { "New tracking", "Continue existing CSV" });
        sessionMode.SelectedIndex = 0;
        sessionMode.Dock = DockStyle.Fill;
        layout.Controls.Add(sessionMode, 1, 3);
        AddPath(layout, 1, "Video", video, PickVideo);
        AddPath(layout, 2, "Config (optional)", config, PickConfig);
        outputBrowse = AddPath(layout, 4, "Track CSV", output, PickOutput);
        layout.Controls.Add(new Label { Text = "Playback speed", Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft }, 0, 5);
        speed.DropDownStyle = ComboBoxStyle.DropDownList;
        speed.Items.AddRange(new object[] { "0.1x", "0.25x", "0.5x", "1x" });
        speed.SelectedIndex = 1;
        speed.Width = 110;
        layout.Controls.Add(speed, 1, 5);

        hint.Dock = DockStyle.Fill;
        layout.Controls.Add(hint, 0, 6);
        layout.SetColumnSpan(hint, 3);

        launch.Dock = DockStyle.Right;
        launch.Width = 160;
        launch.Click += StartTracker;
        layout.Controls.Add(launch, 1, 7);
        status.Text = "Ready";
        status.Dock = DockStyle.Fill;
        status.TextAlign = ContentAlignment.MiddleLeft;
        layout.Controls.Add(status, 0, 8);
        layout.SetColumnSpan(status, 3);
        AcceptButton = launch;
        sessionMode.SelectedIndexChanged += delegate { UpdateSessionMode(); };
        UpdateSessionMode();

        if (args.Length > 0 && File.Exists(args[0]))
            SetVideo(Path.GetFullPath(args[0]));
    }

    private Button AddPath(TableLayoutPanel layout, int row, string label, TextBox field, EventHandler browse)
    {
        layout.Controls.Add(new Label { Text = label, Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft }, 0, row);
        field.Dock = DockStyle.Fill;
        layout.Controls.Add(field, 1, row);
        Button button = new Button { Text = "Browse...", Dock = DockStyle.Fill };
        button.Click += browse;
        layout.Controls.Add(button, 2, row);
        return button;
    }

    private void UpdateSessionMode()
    {
        bool resume = sessionMode.SelectedIndex == 1;
        outputBrowse.Text = resume ? "Open CSV..." : "Save as...";
        launch.Text = resume ? "Continue tracking" : "Open tracker";
        hint.Text = resume
            ? "Use the same video and config. Your marks stay; tracking opens at the first unscored frame."
            : "Choose a new CSV for this session. No config means the full video.";
    }

    private void SetVideo(string path)
    {
        video.Text = path;
        output.Text = NewTrackPath(Path.Combine(repository, "output", "tracks", "manual", Path.GetFileNameWithoutExtension(path) + ".csv"));
        sessionMode.SelectedIndex = 0;
        status.Text = "Ready for new tracking. Select Continue existing CSV to load saved marks.";
    }

    private static string NewTrackPath(string path)
    {
        string candidate = path;
        int suffix = 2;
        while (File.Exists(candidate) || Directory.Exists(candidate))
        {
            candidate = Path.Combine(Path.GetDirectoryName(path),
                Path.GetFileNameWithoutExtension(path) + "_" + suffix + Path.GetExtension(path));
            suffix++;
        }
        return candidate;
    }

    private void PickVideo(object sender, EventArgs args)
    {
        using (OpenFileDialog dialog = new OpenFileDialog())
        {
            dialog.Title = "Choose a video";
            dialog.Filter = "Videos|*.avi;*.mp4;*.mov;*.mkv;*.wmv;*.mpg;*.mpeg;*.m4v|All files|*.*";
            if (dialog.ShowDialog(this) == DialogResult.OK)
                SetVideo(dialog.FileName);
        }
    }

    private void PickConfig(object sender, EventArgs args)
    {
        using (OpenFileDialog dialog = new OpenFileDialog())
        {
            dialog.Title = "Choose the arena and scoring-window config (cancel to keep current)";
            dialog.Filter = "RATEL config|*.toml";
            dialog.InitialDirectory = Path.Combine(repository, "configs");
            if (dialog.ShowDialog(this) == DialogResult.OK)
                config.Text = dialog.FileName;
        }
    }

    private void PickOutput(object sender, EventArgs args)
    {
        if (sessionMode.SelectedIndex == 1)
        {
            using (OpenFileDialog dialog = new OpenFileDialog())
            {
                dialog.Title = "Continue an existing manual track";
                dialog.Filter = "Manual track CSV|*.csv";
                dialog.FileName = output.Text;
                if (dialog.ShowDialog(this) == DialogResult.OK)
                    output.Text = dialog.FileName;
            }
            return;
        }
        using (SaveFileDialog dialog = new SaveFileDialog())
        {
            dialog.Title = "Choose a new manual track CSV";
            dialog.Filter = "Manual track CSV|*.csv";
            dialog.DefaultExt = "csv";
            dialog.AddExtension = true;
            // New sessions use a fresh filename rather than replacing an existing CSV.
            dialog.OverwritePrompt = false;
            dialog.FileName = output.Text;
            if (dialog.ShowDialog(this) == DialogResult.OK)
                output.Text = dialog.FileName;
        }
    }

    private static string Quote(string value)
    {
        // File arguments cannot contain quotes, and each is a file (no trailing slash).
        if (value.IndexOf('"') >= 0 || value.EndsWith("\\"))
            throw new ArgumentException("Please select a file path without quote characters.");
        return "\"" + value + "\"";
    }

    private async void StartTracker(object sender, EventArgs args)
    {
        try
        {
            string videoPath = Path.GetFullPath(video.Text.Trim());
            string outputPath = Path.GetFullPath(output.Text.Trim());
            string configPath = config.Text.Trim();
            if (!File.Exists(videoPath))
                throw new FileNotFoundException("Choose an existing video first.");
            if (configPath.Length > 0)
            {
                configPath = Path.GetFullPath(configPath);
                if (!File.Exists(configPath))
                    throw new FileNotFoundException("The config file does not exist.");
            }
            if (!String.Equals(Path.GetExtension(outputPath), ".csv", StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("Choose a .csv output file.");
            bool resume = sessionMode.SelectedIndex == 1;
            if (resume && !File.Exists(outputPath))
                throw new FileNotFoundException("Choose the existing manual CSV you want to continue.");
            if (!resume)
            {
                outputPath = NewTrackPath(outputPath);
                output.Text = outputPath;
            }
            string arguments = "-m behaviour manual-track " + Quote(videoPath) +
                " --speed " + speed.SelectedItem.ToString().TrimEnd('x') + " -o " + Quote(outputPath);
            if (configPath.Length > 0)
                arguments += " -c " + Quote(configPath);
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = Path.Combine(repository, ".venv-oft", "Scripts", "pythonw.exe");
            start.Arguments = arguments;
            start.WorkingDirectory = repository;
            start.UseShellExecute = false;
            start.CreateNoWindow = true;
            start.RedirectStandardOutput = true;
            start.RedirectStandardError = true;
            launch.Enabled = false;
            status.Text = "Tracker open - marks autosave while you work.";
            string error = await Task.Run(() =>
            {
                using (Process child = new Process())
                {
                    child.StartInfo = start;
                    child.OutputDataReceived += delegate { };
                    child.Start();
                    child.BeginOutputReadLine();
                    string details = child.StandardError.ReadToEnd();
                    child.WaitForExit();
                    return child.ExitCode == 0 ? null : details;
                }
            });
            if (IsDisposed)
                return;
            status.Text = error == null ? "Tracker closed. Ready for another video." : "Could not open tracker.";
            if (error != null)
                MessageBox.Show(this, String.IsNullOrWhiteSpace(error) ? "RATEL exited with an error." : error,
                                "RATEL", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        catch (Exception error)
        {
            if (!IsDisposed)
                MessageBox.Show(this, error.Message, "RATEL", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            if (!IsDisposed)
                launch.Enabled = true;
        }
    }
}
