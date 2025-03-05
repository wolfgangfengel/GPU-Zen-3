using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Hlsl2Python
{
    internal partial class Program
    {
        private static int RunProcess(string fileName, string args, ProcessStartOption option, Stream? istream, Stream? ostream, IList<string>? input, StringBuilder? output, StringBuilder? error, bool redirectOutputToConsole, bool redirectErrorToConsole, Encoding encoding)
        {
            //Considering cross-platform compatibility, do not use specific process environment variables.
            try {
                Process p = new Process();
                var psi = p.StartInfo;
                psi.FileName = fileName;
                psi.Arguments = args;
                psi.UseShellExecute = option.UseShellExecute;
                if (null != option.Verb) {
                    psi.Verb = option.Verb;
                }
                if (null != option.Domain) {
                    if(OperatingSystem.IsWindows())
                        psi.Domain = option.Domain;
                }
                if (null != option.UserName) {
                    psi.UserName = option.UserName;
                }
                if (null != option.Password) {
                    unsafe {
                        fixed (char* pchar = option.Password.ToCharArray()) {
                            if (OperatingSystem.IsWindows())
                                psi.Password = new System.Security.SecureString(pchar, option.Password.Length);
                        }
                    }
                }
                if (null != option.PasswordInClearText) {
                    if (OperatingSystem.IsWindows())
                        psi.PasswordInClearText = option.PasswordInClearText;
                }
                if (OperatingSystem.IsWindows())
                    psi.LoadUserProfile = option.LoadUserProfile;
                psi.WindowStyle = option.WindowStyle;
                psi.CreateNoWindow = !option.NewWindow;
                psi.ErrorDialog = option.ErrorDialog;
                psi.WorkingDirectory = option.WorkingDirectory;

                if (null != istream || null != input) {
                    psi.RedirectStandardInput = true;
                }
                if (null != ostream || null != output || redirectOutputToConsole) {
                    psi.RedirectStandardOutput = true;
                    psi.StandardOutputEncoding = encoding;
                    var tempStringBuilder = new StringBuilder();
                    p.OutputDataReceived += (sender, e) => OnOutputDataReceived(sender, e, ostream, output, redirectOutputToConsole, encoding, tempStringBuilder);
                }
                if (null != error || redirectErrorToConsole) {
                    psi.RedirectStandardError = true;
                    psi.StandardErrorEncoding = encoding;
                    var tempStringBuilder = new StringBuilder();
                    p.ErrorDataReceived += (sender, e) => OnErrorDataReceived(sender, e, ostream, error, redirectErrorToConsole, encoding, tempStringBuilder);
                }
                if (p.Start()) {
                    if (psi.RedirectStandardInput) {
                        if (null != istream) {
                            istream.Seek(0, SeekOrigin.Begin);
                            using (var sr = new StreamReader(istream, encoding, true, 1024, true)) {
                                string? line;
                                while ((line = sr.ReadLine()) != null) {
                                    p.StandardInput.WriteLine(line);
                                    p.StandardInput.Flush();
                                }
                            }
                            p.StandardInput.Close();
                        }
                        else if (null != input) {
                            foreach (var line in input) {
                                p.StandardInput.WriteLine(line);
                                p.StandardInput.Flush();
                            }
                            p.StandardInput.Close();
                        }
                    }
                    if (null != ostream) {
                        ostream.Seek(0, SeekOrigin.Begin);
                        ostream.SetLength(0);
                    }
                    if (psi.RedirectStandardOutput)
                        p.BeginOutputReadLine();
                    if (psi.RedirectStandardError)
                        p.BeginErrorReadLine();
                    p.WaitForExit();
                    if (psi.RedirectStandardOutput) {
                        p.CancelOutputRead();
                    }
                    if (psi.RedirectStandardError) {
                        p.CancelErrorRead();
                    }
                    int r = p.ExitCode;
                    p.Close();
                    return r;
                }
                else {
                    Console.WriteLine("process({0} {1}) failed.", fileName, args);
                    return -1;
                }
            }
            catch (Exception ex) {
                Console.WriteLine("process({0} {1}) exception:{2} stack:{3}", fileName, args, ex.Message, ex.StackTrace);
                while (null != ex.InnerException) {
                    ex = ex.InnerException;
                    Console.WriteLine("\t=> exception:{0} stack:{1}", ex.Message, ex.StackTrace);
                }
                return -1;
            }
        }
        private static void OnOutputDataReceived(object sender, DataReceivedEventArgs e, Stream? ostream, StringBuilder? output, bool redirectToConsole, Encoding encoding, StringBuilder temp)
        {
            var p = sender as Process;
            if (null == p)
                return;
            if (p.StartInfo.RedirectStandardOutput) {
                temp.Length = 0;
                temp.AppendLine(e.Data);
                var txt = temp.ToString();
                if (null != ostream) {
                    var bytes = encoding.GetBytes(txt);
                    ostream.Write(bytes, 0, bytes.Length);
                }
                if (null != output) {
                    output.Append(txt);
                }
                if (redirectToConsole) {
                    Console.Write(txt);
                }
            }
        }
        private static void OnErrorDataReceived(object sender, DataReceivedEventArgs e, Stream? ostream, StringBuilder? error, bool redirectToConsole, Encoding encoding, StringBuilder temp)
        {
            var p = sender as Process;
            if (null == p)
                return;
            if (p.StartInfo.RedirectStandardError) {
                temp.Length = 0;
                temp.AppendLine(e.Data);
                var txt = temp.ToString();
                if (null != ostream) {
                    var bytes = encoding.GetBytes(txt);
                    ostream.Write(bytes, 0, bytes.Length);
                }
                if (null != error) {
                    error.Append(txt);
                }
                if (redirectToConsole) {
                    Console.Write(txt);
                }
            }
        }

        internal sealed class ProcessStartOption
        {
            internal bool UseShellExecute = false;
            internal string? Verb = null;
            internal string? Domain = null;
            internal string? UserName = null;
            internal string? Password = null;
            internal string? PasswordInClearText = null;
            internal bool LoadUserProfile = false;
            internal ProcessWindowStyle WindowStyle = ProcessWindowStyle.Normal;
            internal bool NewWindow = false;
            internal bool ErrorDialog = false;
            internal string WorkingDirectory = Environment.CurrentDirectory;
        }
    }
}