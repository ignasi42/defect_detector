
function [bark_mean, relative_bark] = bark_spectrum_processing(inputFile)

    addpath c:\\Users\ASUS\Desktop\IGNASI\SMC\Workspace\;
	%addpath E:\\Workspace\test_batch\output_riaa_ok\;
	%addpath E:\\Workspace\test_batch\output_riaa_ko\;
	%addpath E:\\Workspace\test_batch\output_riaa_rec\;
    addpath c:\\Program' Files'\MATLAB\R2015b\toolbox\ma\;

    % COMPUTE THE BARK SPECTRUM FOR EACH FILE:
    
    %filename_bark_ma = strcat(inputDir,'\',inputFile,'_bark_ma.jpg');
    filename_bark_ma_mean = strcat(inputFile,'_bark_ma_mean.jpg');

    % Using MA-Toolbox
    [wav, fs] = audioread(inputFile);
    wav_mono = (wav(:,1)+wav(:,2))/2;
    p.fs = fs;
    p.do_visu = 1;
    p.do_sone = 0;
    p.fft_size = 1024;
    [bark, Ntot, p] = ma_sone(wav_mono,p);

    bark_mean = mean(bark,2);   		% Average for each band along the frames.
    acum_bark = sum(bark_mean.^2);      % Summing energy of all bark bands.

    relative_bark = zeros(size(bark_mean));

    for i=1 : size(bark_mean)
        %energy = bark_mean(i).^2;
        relative_bark(i) = (bark_mean(i).^2/acum_bark)*100;   % Calculating percentage of each band.
    end

    subplot(1,2,1);
    plot(bark_mean);
    title('Mean Bark-Spectrum');
    xlabel('Bark bands');
    ylabel('Magnitude [dB]');
    subplot(1,2,2);
    plot(relative_bark);
    title('Percentage Mean Bark-Spectrum');
    xlabel('Bark bands');
    ylabel('Energy [%]');
    print(filename_bark_ma_mean,'-djpeg');
    close;

end
